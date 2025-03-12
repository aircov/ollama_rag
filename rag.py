# -*- coding: utf-8 -*-
# @Time    : 2025/2/24 17:30
# @Author  : yaomw
# @Desc    : rag全流程

import logging
from pathlib import Path

import psutil
import faiss
import hashlib
import os
from typing import List
from elasticsearch import Elasticsearch, helpers

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore import InMemoryDocstore
from langchain_elasticsearch import ElasticsearchRetriever

from config import EsUrl, EsIndexName, OllamaUrl, OllamaModelName, EmbeddingDim
from embed.corom_embeddings import CoROMEmbeddings
from es.dsl import dsl
from rerank.corom_rerank import CoROMRerank


class RAGPipeline:
    def __init__(self, model_name: str = "deepseek-r1:14b", max_memory_gb: float = 3.0):
        self.ollama_url = OllamaUrl
        self.setup_logging()
        self.check_system_memory(max_memory_gb)
        
        # 加载deepseek-r1模型
        self.llm = OllamaLLM(
            base_url=self.ollama_url,  # 本地服务地址（必须显式指定）
            model=model_name,  # 本地已下载的模型名称
            temperature=0.8,  # 可选参数控制生成随机性
            num_ctx=10000  # 增加上下文窗口
        )
        
        # 使用自定义的本地嵌入模型
        self.embeddings = CoROMEmbeddings()
        
        self.rerank = CoROMRerank()
        
        # 配置 Elasticsearch 连接
        self.es_url = EsUrl
        self.es_client = Elasticsearch(self.es_url)
        if not self.es_client.ping():
            self.logger.error("无法连接到 Elasticsearch 实例")
        else:
            self.logger.info("成功连接到 Elasticsearch 实例")
        
        self.es_index_name = EsIndexName
        
        # 中文版prompt
        self.prompt = ChatPromptTemplate.from_template("""
        请根据以下上下文详细且有条理地回答用户的问题。如果在提供的信息中找不到答案，或者提供的上下文与问题不相关，请基于你的知识库生成合理的答案，
        确保答案逻辑清晰且符合常识，此时不需要结合上下文判断，直接回答用户基础问题即可。

        使用中文回答，并保持语言简洁、易懂。

        上下文: {context}
        问题: {question}
        回答: """)
    
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")
    
    def load_and_split_documents(self, file_path: str) -> List[Document]:
        """拆分文档片段"""
        data = []
        for filename in os.listdir(file_path):
            print(f"加载文件：{filename}")
            # txt
            if filename.endswith(".txt"):
                loader = TextLoader(f"{file_path}{filename}", encoding="utf-8")
                documents = loader.load()
                documents = self.normalize_source(documents, filename)
                data.extend(documents)
            
            # pdf
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(f"{file_path}{filename}", extract_images=False)
                documents = loader.load()
                documents = self.normalize_source(documents, filename)
                data.extend(documents)
            
            # Markdown文件处理
            elif filename.endswith(".md"):
                headers_to_split_on = [
                    ("#", "H1"),
                    ("##", "H2"),
                    ("###", "H3"),
                    ("####", "H4"),
                    ("#####", "H5"),
                ]
                markdown_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=headers_to_split_on,
                    return_each_line=False,
                    strip_headers=True  # 去除标题中的HTML标签
                )
                text = Path(f"{file_path}{filename}").read_text(encoding="utf-8")
                splits = markdown_splitter.split_text(text)
                
                splits = self.normalize_source(splits, filename)
                data.extend(splits)
            
            # HTML文件处理
            elif filename.endswith(".html"):
                headers_to_split_on = [
                    ("h1", "H1"),
                    ("h2", "H2"),
                    ("h3", "H3"),
                    ("h4", "H4"),
                    ("h5", "H5"),
                ]
                html_splitter = HTMLHeaderTextSplitter(
                    headers_to_split_on=headers_to_split_on
                )
                splits = html_splitter.split_text_from_file(file_path)
                splits = self.normalize_source(splits, filename)
                data.extend(splits)
        
        # self.logger.info(f"加载数据格式样例：{data[0]}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # 分隔文档的块大小
            chunk_overlap=80,  # 分隔文档的块重叠大小，保留上下文语义
            length_function=len,  # 计算文本长度的函数
            add_start_index=True,  # 是否添加起始索引
            separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]  # 按自然语言结构分割
        )
        splits = text_splitter.split_documents(data)
        self.logger.info(f"split doc:{splits[-1]}")
        self.logger.info(f"Created {len(splits)} document chunks")
        return splits
    
    def normalize_source(self, docs: List[Document], filename: str) -> List[Document]:
        """统一source字段为纯文件名"""
        for doc in docs:
            doc.metadata["source"] = filename
        return docs
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """创建带批处理的向量存储"""
        dim = EmbeddingDim
        
        # 创建使用内积（余弦相似度）的索引
        index = faiss.IndexFlatIP(dim)  # IP表示Inner Product
        # 初始化FAISS时指定度量方式
        vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            normalize_L2=True
        )
        
        batch_size = 8
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(batch)
            self.logger.info(f"Added batch {i // batch_size + 1}")
        
        self.logger.info(f"向量数据库构建完成！！")
        return vectorstore
    
    def create_es_retriever(self, documents: List[Document]) -> ElasticsearchRetriever:
        """创建Elasticsearch检索器"""
        
        self.es_client.indices.create(index=self.es_index_name, body=dsl["create_index"], ignore=400)
        
        # 将文档添加到es
        actions = [
            {
                "_index": self.es_index_name,
                "_id": hashlib.md5(doc.page_content.encode('utf-8')).hexdigest(),
                "_source": {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": self.embeddings.embed_query(doc.page_content)
                }
            }
            for doc in documents
        ]
        helpers.bulk(self.es_client, actions, chunk_size=200)
        self.es_client.indices.refresh(index=self.es_index_name)
        result = self.es_client.count(index=self.es_index_name)
        self.logger.info(f"ES索引 {self.es_index_name} 中有 {result['count']} 条记录")
        
        def default_body_func(query: str) -> dict:
            # 定义 body_func：生成 Elasticsearch 查询体
            return {
                "_source": ["content", "metadata"],
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "analyzer": "ik_max_word_synonym"
                        }
                    }
                },
                "knn": {
                    "field": "embedding",
                    "k": 5,
                    "num_candidates": 100,
                    "query_vector": self.embeddings.embed_query(query),
                    "boost": 1
                },
                "size": 20
            }
        
        retriever = ElasticsearchRetriever(
            es_client=self.es_client,
            index_name=self.es_index_name,
            content_field="content",
            body_func=default_body_func
        )
        self.logger.info(f"创建 Elasticsearch 检索器完成！！")
        
        return retriever
    
    def setup_rag_chain(self, es_retriever: ElasticsearchRetriever):
        # 构建rag链
        
        def combined_retriever(question: str) -> list:
            # es 文本检索 + 向量检索
            es_search_docs = es_retriever.invoke(question)
            # self.logger.info(f"es 搜索：{es_search_docs}")
            
            # 重新排序
            rerank_docs = self.rerank_results(question, es_search_docs, method="corom", top_k=5)
            
            print("\n=== 多路召回再重排 ===")
            for i, doc in enumerate(rerank_docs, 1):
                print(
                    f"【文档{i} | {doc.metadata.get('source', '')} | rerank得分: {doc.metadata.get('rerank_score',0):.4f}】\n{doc}\n{'-' * 50}")
            
            return rerank_docs
        
        printable_retriever = RunnableLambda(combined_retriever)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # rag链
        rag_chain = (
            # 步骤1：组装输入数据
                {
                    "context": printable_retriever | format_docs,  # 先检索文档再格式化
                    "question": RunnablePassthrough()  # 直接传递原始问题
                }
                # 步骤2：组合提示词
                | self.prompt
                # 步骤3：调用大模型
                | self.llm
                # 步骤4：解析输出
                | StrOutputParser()
        )
        
        return rag_chain
    
    def query(self, chain, question: str) -> str:
        """执行问答"""
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        # return chain.invoke(question)
        return chain.stream(question)
    
    def rerank_results(self, question, docs, method="corom", top_k=5):
        """
        对检索结果进行重排序
        :param question:查询字符串
        :param docs:文档内容列表
        :param method:重排序方法 ("cross_encoder", "llm" 或 None)
        :param top_k:返回结果数量
        :return:
        """
        if method == "corom":
            return self.rerank_documents_with_corom(question, docs, top_k)
        elif method == "llm":
            return self.rerank_documents_with_llm(question, docs, top_k)
        else:
            # 使用默认方法（即不进行重排序）
            return docs
    
    def rerank_documents_with_corom(self, question: str, es_search_docs: list, top_k: int) -> list:
        """重排，使用魔塔模型"""
        content_list = [doc.page_content for doc in es_search_docs]
        doc_rerank = self.rerank.rerank(question, content_list)
        for idx, score in enumerate(doc_rerank):
            es_search_docs[idx].metadata["rerank_score"] = score['score']
        
        # 重新排序
        ret = sorted(es_search_docs, key=lambda x: x.metadata.get("rerank_score"), reverse=True)
        
        ret = [doc for doc in ret if doc.metadata.get("rerank_score", 0) >= 0.1][:top_k]
        
        return ret
    
    def rerank_documents_with_llm(self, question, docs, top_k):
        """
        使用LLM对检索结果进行重排序
        :param question:
        :param docs:
        :param top_k:
        :return:
        """
        try:
            content_list = [doc.page_content for doc in docs]
            prompt = f"""给定以下查询和文档片段，评估它们的相关性。
            文档是一个列表有多个片段，你需要分析每一个片段，然后进行打分，返回每一个片段的分数，一一对应，以列表格式返回。
            请不要解释你的评分，只返回一个列表，列表长度与文档片段数量相同，列表中的每个元素都是0-10之间的整数。
            评分标准：0分表示完全不相关，10分表示高度相关。
            只需返回0-10之间的整数分数，不要有任何其他解释。
    
            查询: {question}
    
            文档片段: {content_list}
            
            文档片段列表长度: {len(content_list)}
    
            相关性分数(0-10):"""
            
            resp = self.llm.invoke(prompt)
            
            llm_score = resp.split("\n\n")[-1].replace("'", "")
            self.logger.info(f"LLM 评分：{llm_score}")
            llm_score = eval(llm_score)
            if len(llm_score) != len(content_list):
                self.logger.error(f"LLM 评分长度不匹配：{len(llm_score)} != {len(content_list)}")
                return docs[:top_k]
            for idx, score in enumerate(llm_score):
                docs[idx].metadata["rerank_score"] = score
            
            return sorted(docs, key=lambda x: x.metadata.get("rerank_score"), reverse=True)[:top_k]
        
        except Exception as e:
            self.logger.error(f"llm rerank Error: {e}")
            return docs[:top_k]


if __name__ == '__main__':
    
    rag = RAGPipeline(model_name=OllamaModelName, max_memory_gb=3.0)
    documents = rag.load_and_split_documents("./data/")
    es_retriever = rag.create_es_retriever(documents=documents)
    
    chain = rag.setup_rag_chain(es_retriever)
    
    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        
        # question = "神烔?"
        # question = "盗龄医生?"
        print(f"Question: {question}\nAnswer: ", end='', flush=True)
        for chunk in rag.query(chain, question):
            print(chunk, end="", flush=True)
