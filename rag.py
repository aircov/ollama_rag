# -*- coding: utf-8 -*-
# @Time    : 2025/2/24 17:30
# @Author  : yaomw
# @Desc    : rag全流程

from datetime import datetime
from pathlib import Path

import psutil
import hashlib
import os
from typing import List, Tuple
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
from langchain_elasticsearch import ElasticsearchRetriever

from config import EsUrl, EsIndexName, OllamaUrl, OllamaModelName, EmbeddingDim
from embed.corom_embeddings import CoROMEmbeddings
from es.dsl import dsl
from recall.serper_client import SerperClient
from recall.web_search import google_search
from rerank.corom_rerank import CoROMRerank
from rerank.llm_rerank import LLMRerank
from utils.utils import get_file_list
from extensions import logger


class RAGPipeline:
    def __init__(self, model_name: str = "deepseek-r1:14b", max_memory_gb: float = 3.0):
        self.ollama_url = OllamaUrl
        self.logger = logger
        self.check_system_memory(max_memory_gb)
        
        # 加载deepseek-r1模型
        self.llm = OllamaLLM(
            base_url=self.ollama_url,  # 本地服务地址（必须显式指定）
            model=model_name,  # 本地已下载的模型名称
            temperature=0.8,  # 可选参数控制生成随机性
            num_ctx=64000  # 增加上下文窗口
        )
        
        # 使用自定义的本地嵌入模型
        self.embeddings = CoROMEmbeddings()
        
        self.search_client = SerperClient()
        
        self.corom_rerank = CoROMRerank()
        self.llm_rerank = LLMRerank(self.llm)
        
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
        请根据以下上下文和对话历史，结合大语言模型的知识储备，详细且有条理地回答用户的问题。
        
        如果在提供的信息中找不到答案，或者提供的上下文与问题不相关，请基于你的知识库生成合理的答案，确保答案逻辑清晰且符合常识，
        此时不需要结合上下文判断，直接回答用户基础问题即可。不要回答上下文的内容。
        
        当前时间是：{current_time}
        
        对话历史（最多{max_turns}轮）：
        {history}

        使用中文回答，并保持语言简洁、易懂。

        上下文: {context}
        问题: {question}
        回答: """)
    
    def check_system_memory(self, max_memory_gb: float):
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        self.logger.info(f"Available system memory: {available_memory:.1f} GB")
        if available_memory < max_memory_gb:
            self.logger.warning("Memory is below recommended threshold.")
    
    def load_and_split_documents(self, file_path="", files=None):
        """拆分文档片段"""
        data = []
        
        if file_path:
            file_names = os.listdir(file_path)
        
        if files:
            file_names = [os.path.basename(i) for i in files]
            file_path = os.path.dirname(files[0]) + "/"
        
        for filename in file_names:
            print(f"加载文件：{filename}")
            # txt
            if filename.endswith(".txt") or filename.endswith(".doc") or filename.endswith(".docx"):
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
        self.logger.info(f"Created {len(splits)} document chunks")
        
        # 添加索引
        self.update_es_index(documents=splits)
        
        # 获取更新后的文件列表
        file_list = get_file_list(self.es_client, self.es_index_name)
        
        self.logger.info(f"已索引文件列表：{file_list}")
        
        return file_list
    
    def normalize_source(self, docs: List[Document], filename: str) -> List[Document]:
        """统一source字段为纯文件名"""
        for doc in docs:
            doc.metadata["source"] = filename
        return docs
    
    def update_es_index(self, documents: List[Document]):
        """数据插入es"""
        
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
    
    def get_es_retriever(self):
        """获取Elasticsearch检索器"""
        
        def default_body_func(query: str) -> dict:
            # 定义 body_func：生成 Elasticsearch 查询体
            # 线性加权，也可以使用倒序排序融合rrf
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
    
    def setup_rag_chain(self, rerank_method="corom", enable_web_search=False, max_turns=20):
        # 构建rag链
        es_retriever = self.get_es_retriever()
        
        def combined_retriever(question: str) -> list:
            # 如果启用网页搜索，则完全跳过 ES 检索
            if enable_web_search:
                # 启用网页搜索
                try:
                    self.logger.info(f"开始网络搜索：{question}")
                    # online_search_results = google_search(question, 20)
                    online_search_results = self.search_client.search_sync(question)
                    
                    web_search_docs = [
                        Document(
                            page_content=result["content"],  # 使用 description 作为内容
                            metadata={
                                "source": result["url"],
                                "title": result["title"],
                                "rerank_score": 0.0  # 初始化重排分数
                            }
                        ) for result in online_search_results
                    ]
                    self.logger.info(f"网页搜索数据：{len(web_search_docs)}条")
                    self.logger.info(f"网页搜索数据详情：{online_search_results}")
                    
                    return self.rerank_results(question, web_search_docs, method="", top_k=5)
                
                except Exception as e:
                    self.logger.error(f"Google搜索发生错误：{e}")
                    return []
            
            else:
                # es 文本检索 + 向量检索
                es_search_docs = es_retriever.invoke(question)
                self.logger.info(f"es 搜索数据：{len(es_search_docs)}条")
                
                # 重新排序
                rerank_docs = self.rerank_results(question, es_search_docs, method=rerank_method, top_k=5)
                
                # print("\n=== 多路召回再重排 ===")
                # for i, doc in enumerate(rerank_docs, 1):
                #     print(
                #         f"【文档{i} | {doc.metadata.get('source', '')} | rerank得分: {doc.metadata.get('rerank_score', 0):.4f}】\n{doc}\n{'-' * 50}")
                
                return rerank_docs
        
        printable_retriever = RunnableLambda(lambda x: combined_retriever(x["question"]))
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 新增历史处理逻辑
        def format_history(history: List[Tuple], max_turns: int) -> str:
            """将历史记录格式化为文本"""
            
            if not history:
                return "暂无对话历史"
            
            # 保留最近max_turns轮对话
            trimmed_history = history[-max_turns:]
            
            history_str = []
            for i, (user_q, ai_a) in enumerate(trimmed_history, 1):
                history_str.append(
                    f"第{i}轮对话：\n"
                    f"问：{user_q}\n"
                    f"答：{ai_a}\n"
                )
            
            s = "\n".join(history_str) if history_str else "无相关对话历史"
            self.logger.info(f"历史对话：{s}")
            return s
        
        # rag链
        rag_chain = (
            # 步骤1：组装输入数据
                {
                    "context": printable_retriever | format_docs,  # 先检索文档再格式化
                    "question": RunnablePassthrough(),  # 直接传递原始问题
                    "current_time": RunnableLambda(lambda _: datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")),
                    "history": RunnableLambda(lambda x: format_history(x["history"], max_turns)),
                    "max_turns": RunnableLambda(lambda _: max_turns)
                }
                # 步骤2：组合提示词
                | self.prompt
                # 步骤3：调用大模型
                | self.llm
                # 步骤4：解析输出
                | StrOutputParser()
        )
        
        return rag_chain
    
    def query(self, chain, question: str, history: list = None) -> str:
        """执行问答"""
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        self.logger.info(f"Memory usage: {memory_usage:.1f} MB")
        
        input_data = {
            "question": question,
            "history": history or []
        }
        
        # return chain.invoke(question)
        return chain.stream(input_data)
    
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
            return self.corom_rerank.rerank_documents_with_corom(question, docs, top_k)
        elif method == "llm":
            return self.llm_rerank.rerank_documents_with_llm(question, docs, top_k)
        else:
            # 使用默认方法（即不进行重排序）
            return docs


if __name__ == '__main__':
    
    rag = RAGPipeline(model_name=OllamaModelName, max_memory_gb=3.0)
    rag.load_and_split_documents("./data/")
    
    chain = rag.setup_rag_chain("corom", enable_web_search=False)
    
    # 模拟多轮对话
    conversation_history = []
    
    while True:
        question = input("请输入问题：")
        full_answer = ""
        conversation_history.append((question, ""))  # 新增一轮对话，初始回答为空
        
        if question == "exit":
            break
        
        # question = "神烔?"
        # question = "盗龄医生?"
        # question = "解方程 (x²-5x+6=0)。"
        print(f"Question: {question}\nAnswer: ", end='', flush=True)
        for chunk in rag.query(chain, question, history=conversation_history):
            print(chunk, end="", flush=True)
            full_answer += chunk
            
        conversation_history[-1] = (question, full_answer)
            
