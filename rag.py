# -*- coding: utf-8 -*-
# @Time    : 2025/2/24 17:30
# @Author  : yaomw
# @Desc    : rag全流程

import logging
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
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.docstore import InMemoryDocstore
from langchain_elasticsearch import ElasticsearchRetriever

from config import EsUrl, EsIndexName, OllamaUrl, OllamaModelName
from ollama_embeddings import OllamaEmbeddings
from es.dsl import dsl


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
        self.embeddings = OllamaEmbeddings(
            base_url=self.ollama_url,
            model=model_name  # 使用同一个模型
        )
        
        # 配置 Elasticsearch 连接
        self.es_url = EsUrl
        self.es_client = Elasticsearch(self.es_url)
        if not self.es_client.ping():
            self.logger.error("无法连接到 Elasticsearch 实例")
        else:
            self.logger.info("成功连接到 Elasticsearch 实例")
        
        self.es_index_name = EsIndexName
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context. Be concise.
        If you cannot find the answer in the context, say "I cannot answer this based on the provided context."

        Context: {context}
        Question: {question}
        Answer: """)
        
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
            if filename.endswith(".txt"):
                loader = TextLoader(f"{file_path}{filename}", encoding="utf-8")
                documents = loader.load()
                data.extend(documents)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(f"{file_path}{filename}", extract_images=True)
                documents = loader.load()
                data.extend(documents)
        
        # self.logger.info(f"加载数据格式样例：{data[0]}")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # 分隔文档的块大小
            chunk_overlap=80,  # 分隔文档的块重叠大小，保留上下文语义
            length_function=len,  # 计算文本长度的函数
            add_start_index=True  # 是否添加起始索引
        )
        splits = text_splitter.split_documents(data)
        self.logger.info(f"split doc:{splits[-1]}")
        self.logger.info(f"Created {len(splits)} document chunks")
        return splits
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """创建带批处理的向量存储"""
        dim = 5120
        
        # 创建使用内积（余弦相似度）的索引
        index = faiss.IndexFlatIP(dim)  # IP表示Inner Product
        # 初始化FAISS时指定度量方式
        vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            normalize_L2=False
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
        
        # 将文档添加到 Elasticsearch 索引
        actions = [
            {
                "_index": self.es_index_name,
                "_id": hashlib.md5(doc.page_content.encode('utf-8')).hexdigest(),
                "_source": {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
            }
            for doc in documents
        ]
        helpers.bulk(self.es_client, actions, chunk_size=200)
        self.es_client.indices.refresh(index=self.es_index_name)
        result = self.es_client.count(index=self.es_index_name)
        self.logger.info(f"ES索引 {self.es_index_name} 中有 {result['count']} 条记录")
        
        # 定义 body_func：生成 Elasticsearch 查询体
        def default_body_func(query: str) -> dict:
            return {
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "analyzer": "ik_max_word_synonym"
                        }
                    }
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
    
    def rerank_documents(self, vector_search_docs: list, es_search_docs: list) -> list:
        """使用RRF（排序检索融合）重新排序文档"""
        
        # 用字典保存每个文档的排名信息
        doc_ranks = {}
        id_to_doc = {}
        
        # 向量检索：按向量相似度对文档进行排名
        for rank, (doc, score) in enumerate(vector_search_docs, start=1):
            doc_id = (doc.metadata["source"], doc.metadata["start_index"])
            id_to_doc[doc_id] = doc
            doc_ranks[doc_id] = doc_ranks.get(doc_id, [])  # 确保文档在字典中存在
            doc_ranks[doc_id].append(("vector", rank))
        
        # BM25检索：按BM25得分对文档进行排名
        for rank, doc in enumerate(es_search_docs, start=1):
            doc.metadata = doc.metadata["_source"]["metadata"]
            doc_id = (doc.metadata["source"], doc.metadata["start_index"])
            id_to_doc[doc_id] = doc
            doc_ranks[doc_id] = doc_ranks.get(doc_id, [])
            doc_ranks[doc_id].append(("bm25", rank))
        
        # print(doc_ranks)
        # {('./data/神烔.txt', 672): [('vector', 1), ('bm25', 1)], ('./data/神烔.txt', 305): [('vector', 2)], ('./data/神烔.txt', 0): [('vector', 3), ('bm25', 2)]}
        
        # 使用RRF公式进行得分计算：sum(1 / (k + rank))
        reranked_docs = []
        k = 60  # RRF公式中的常数k
        for doc_id, ranks in doc_ranks.items():
            final_score = 0
            for method, rank in ranks:
                final_score += 1 / (k + rank)  # RRF得分计算（基于排名位置）
            reranked_docs.append((id_to_doc[doc_id], final_score))
        
        # 根据最终得分降序排序文档
        reranked_docs = sorted(reranked_docs, key=lambda x: x[1], reverse=True)
        
        return reranked_docs
    
    def setup_rag_chain(self, vectorstore: FAISS, es_retriever: ElasticsearchRetriever):
        # 构建rag链
        
        def combined_retriever(question: str) -> list:
            # 向量检索
            embedded_q = self.embeddings.embed_query(question)
            vector_search_docs = vectorstore.similarity_search_with_score_by_vector(
                embedded_q,
                k=10,
                fetch_k=20,
                score_threshold=0.35  # (余弦相似度范围 [-1, 1])
            )
            self.logger.info(f"vector 搜索：{vector_search_docs}")
            
            # es 文本检索
            es_search_docs = es_retriever.invoke(question)
            self.logger.info(f"es 搜索：{es_search_docs}")
            
            # 合并两个检索结果
            # all_docs = vector_search_docs + [(doc, score) for doc, score in bm25_docs]
            
            # 使用RRF进行重新排序
            reranked_docs = self.rerank_documents(vector_search_docs, es_search_docs)
            print("\n=== 使用RRF融合（向量 + BM25）后的检索结果 ===")
            for i, (doc, score) in enumerate(reranked_docs, 1):
                print(f"【文档{i} | {doc.metadata.get('source', '')} | RRF得分: {score:.4f}】\n{doc}\n{'-' * 50}")
            
            return [doc for doc, score in reranked_docs]
        
        printable_retriever = RunnableLambda(combined_retriever)
        
        # 单独向量查询
        # def wrapped_retriever(question: str) -> list:
        #     # 使用带分数的搜索方法
        #     embedded_q = self.embeddings.embed_query(question)
        #     # print(f"向量范数：{np.linalg.norm(embedded_q):.4f}（应为1.0）")  # 应输出≈1.0
        #     vector_search_docs = vectorstore.similarity_search_with_score_by_vector(
        #         embedded_q,
        #         k=10,
        #         fetch_k=20,
        #         score_threshold=0.35  # （余弦相似度范围[-1,1]）
        #     )
        #     # print("vector_search_docs",vector_search_docs)
        #
        #     print("\n=== 检索到的内容（带分数） ===")
        #     for i, (doc, score) in enumerate(vector_search_docs, 1):
        #         print(f"【文档{i} | {doc.metadata.get('source', '')} | 相似度分数: {score:.4f}】\n{doc}"
        #               f"\n{'-' * 50}")
        #
        #     return [doc for doc, score in vector_search_docs]
        #
        # printable_retriever = RunnableLambda(wrapped_retriever)
        
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


if __name__ == '__main__':
    
    rag = RAGPipeline(model_name=OllamaModelName, max_memory_gb=3.0)
    documents = rag.load_and_split_documents("./data/")
    vectorstore = rag.create_vectorstore(documents)
    es_retriever = rag.create_es_retriever(documents=documents)
    
    chain = rag.setup_rag_chain(vectorstore, es_retriever)
    
    question = "神烔?"
    print(f"Question: {question}\nAnswer: ", end='', flush=True)
    for chunk in rag.query(chain, question):
        print(chunk, end="", flush=True)
