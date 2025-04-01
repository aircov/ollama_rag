# -*- coding: utf-8 -*-
# @Time    : 2025/2/24 17:30
# @Author  : yaomw
# @Desc    : RAG 全流程（改进版：手写组合确保传入 prompt 的为 mapping）异步

import asyncio
from datetime import datetime
from pathlib import Path
import os
import hashlib
from typing import List, Tuple, AsyncGenerator

from elasticsearch import AsyncElasticsearch, helpers
from langchain_elasticsearch import AsyncElasticsearchRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)
from ollama import AsyncClient

from config import EsUrl, EsIndexName, OllamaUrl, OllamaModelName
from embed.corom_embeddings import CoROMEmbeddings
from es.dsl import dsl
from recall.serper_client import SerperClient
from rerank.corom_rerank import CoROMRerank
from rerank.llm_rerank import LLMRerank
from utils.utils import get_file_list
from extensions import logger


def format_history(history: List[Tuple], max_turns: int = 20) -> str:
    """同步格式化历史对话"""
    if not history:
        return "暂无对话历史"
    
    trimmed = history[-max_turns:]
    
    return "\n".join(
        f"第{i}轮对话：\n"
        f"问：{q}\n"
        f"答：{a.replace('<think>', '<思考中>').replace('</think>', '<思考完成>')}\n"
        for i, (q, a) in enumerate(trimmed, 1)
    )


class AsyncRAGPipeline:
    def __init__(self):
        self.ollama_url = OllamaUrl
        self.logger = logger
        
        # 初始化 Ollama 异步客户端
        self.async_ollama_client = AsyncClient(host=self.ollama_url)
        # 使用自定义的嵌入模型
        self.embeddings = CoROMEmbeddings()
        self.search_client = SerperClient()
        self.corom_rerank = CoROMRerank()
        self.llm_rerank = LLMRerank(self.async_ollama_client)
        
        # 配置 Elasticsearch 连接
        self.es_url = EsUrl
        self.es_client = AsyncElasticsearch(self.es_url)
        if not self.es_client.ping():
            self.logger.error("无法连接到 Elasticsearch 实例")
        else:
            self.logger.info("成功连接到 Elasticsearch 实例")
        self.es_index_name = EsIndexName
        
        # 定义中文 prompt 模板
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
    
    async def load_and_split_documents(self, file_path="", files=None):
        """加载并拆分文档片段"""
        data = []
        if file_path:
            file_names = os.listdir(file_path)
        if files:
            file_names = [os.path.basename(i) for i in files]
            file_path = os.path.dirname(files[0]) + "/"
        
        for filename in file_names:
            print(f"加载文件：{filename}")
            if filename.endswith((".txt", ".doc", ".docx")):
                loader = TextLoader(f"{file_path}{filename}", encoding="utf-8")
                documents = loader.load()
                documents = self.normalize_source(documents, filename)
                data.extend(documents)
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(f"{file_path}{filename}", extract_images=False)
                documents = loader.load()
                documents = self.normalize_source(documents, filename)
                data.extend(documents)
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
                    strip_headers=True
                )
                text = Path(f"{file_path}{filename}").read_text(encoding="utf-8")
                splits = markdown_splitter.split_text(text)
                splits = self.normalize_source(splits, filename)
                data.extend(splits)
            elif filename.endswith(".html"):
                headers_to_split_on = [
                    ("h1", "H1"),
                    ("h2", "H2"),
                    ("h3", "H3"),
                    ("h4", "H4"),
                    ("h5", "H5"),
                ]
                html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                splits = html_splitter.split_text_from_file(file_path)
                splits = self.normalize_source(splits, filename)
                data.extend(splits)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
        )
        splits = text_splitter.split_documents(data)
        self.logger.info(f"Created {len(splits)} document chunks")
        await self.update_es_index(splits)
        file_list = await get_file_list(self.es_client, self.es_index_name)
        self.logger.info(f"已索引文件列表：{file_list}")
        return file_list
    
    def normalize_source(self, docs: List[Document], filename: str) -> List[Document]:
        """统一 source 字段为纯文件名"""
        for doc in docs:
            doc.metadata["source"] = filename
        return docs
    
    async def update_es_index(self, documents: List[Document]):
        """将文档插入到 Elasticsearch 中"""
        await self.es_client.indices.create(index=self.es_index_name, body=dsl["create_index"], ignore=400)
        
        actions = [
            {
                "_index": self.es_index_name,
                "_id": hashlib.md5(doc.page_content.encode("utf-8")).hexdigest(),
                "_source": {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": self.embeddings.embed_query(doc.page_content)
                }
            }
            for doc in documents
        ]
        success, failed = await helpers.async_bulk(self.es_client, actions, chunk_size=200)
        
        await self.es_client.indices.refresh(index=self.es_index_name)
        result = await self.es_client.count(index=self.es_index_name)
        self.logger.info(
            f"ES异步写入完成 | 成功: {success} 失败: {failed} "
            f"总文档数: {result['count']}"
        )
    
    async def async_es_retrieve(self, question: str) -> list:
        """异步执行 Elasticsearch 检索"""
        try:
            from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
            # 获取异步检索器
            retriever = self.get_es_retriever()
            
            # 直接调用异步API
            run_manager = AsyncCallbackManagerForRetrieverRun.get_noop_manager()  # 创建默认管理器
            docs = await retriever._aget_relevant_documents(question, run_manager=run_manager)
            
            self.logger.debug(f"异步检索完成，获得 {len(docs)} 条结果")
            return docs
        except Exception as e:
            self.logger.error(f"ES异步检索失败: {str(e)}")
            return []
    
    def get_es_retriever(self):
        """构建 Elasticsearch 检索器"""
        
        def default_body_func(query: str) -> dict:
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
        
        retriever = AsyncElasticsearchRetriever(
            es_client=self.es_client,
            index_name=self.es_index_name,
            content_field="content",
            body_func=default_body_func
        )
        self.logger.info("创建 Elasticsearch 检索器完成！！")
        return retriever
    
    async def async_combined_retriever(self, rerank_method, question, enable_web_search) -> list:
        """组合检索器：支持 ES 检索和网络搜索"""
        if enable_web_search:
            try:
                self.logger.info(f"开始异步网络搜索：{question}")
                online_results = await self.search_client.search(question)
                web_docs = [
                    Document(
                        page_content=result["content"],
                        metadata={
                            "source": result["url"],
                            "title": result["title"],
                            "rerank_score": 0.0
                        }
                    )
                    for result in online_results
                ]
                return self.rerank_results(question, web_docs, method="", top_k=5)
            except Exception as e:
                self.logger.error(f"异步搜索错误：{e}")
                return []
        else:
            es_docs = await self.async_es_retrieve(question)
            self.logger.info(f"es 搜索数据：{len(es_docs)}条")
            return self.rerank_results(question, es_docs, method=rerank_method, top_k=5)
    
    def rerank_results(self, question, docs, method="corom", top_k=5):
        """对检索结果进行重排序"""
        if method == "corom":
            return self.corom_rerank.rerank_documents_with_corom(question, docs, top_k)
        elif method == "llm":
            return self.llm_rerank.rerank_documents_with_llm(question, docs, top_k)
        else:
            return docs
    
    async def generate_answer(self, question: str, history: List[Tuple] = None,
                              rerank_method: str = "corom", enable_web_search: bool = False,
                              max_turns: int = 20, model_name: str = "deepseek-r1:14b") -> AsyncGenerator[str, None]:
        """
        生成回答的完整流程：
          1. 调用检索器获取上下文文档
          2. 格式化上下文和历史生成 prompt 的输入映射（确保为普通 dict）
          3. 调用 ChatPromptTemplate 的 ainvoke 方法生成最终 prompt
          4. 异步调用 Ollama 模型进行流式回答
        """
        # 获取上下文文档并格式化为字符串
        docs = await self.async_combined_retriever(rerank_method, question, enable_web_search)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        history_str = format_history(history or [], max_turns)
        
        prompt_input = {
            "context": context,
            "question": question,
            "current_time": current_time,
            "history": history_str,
            "max_turns": max_turns
        }
        
        # 确保 prompt_input 为普通字典
        self.logger.debug(f"生成 prompt 输入映射：{prompt_input}")
        
        formatted_prompt = await self.prompt.ainvoke(prompt_input)
        messages = [{"role": "user", "content": msg.content} for msg in formatted_prompt.to_messages()]
        
        try:
            response = await self.async_ollama_client.chat(
                model=model_name,
                messages=messages,
                stream=True
            )
            async for chunk in response:
                yield chunk["message"]["content"]
        except Exception as e:
            self.logger.error(f"模型调用失败: {e}")
            yield "系统处理请求时发生错误"


# 异步流式调用示例
async def main():
    pipeline = AsyncRAGPipeline()
    # 加载和拆分文档（同步方法在独立线程中调用）
    await pipeline.load_and_split_documents("./data/", None)
    
    # 模拟并发请求
    async def concurrent_query(query, history):
        async for chunk in pipeline.generate_answer(query, history, "corom", False, model_name=OllamaModelName):
            print(f"[{query[:10]}...] {chunk}", end="", flush=True)
        print()  # 换行
    
    tasks = [
        concurrent_query("解方程 (x²-5x+6=0)", []),
        concurrent_query("Python的GIL是什么？", []),
        concurrent_query("如何实现快速排序？", [])
    ]
    await asyncio.gather(*tasks)


async def chat_with_history():
    pipeline = AsyncRAGPipeline()
    # 加载和拆分文档（同步方法在独立线程中调用）
    await pipeline.load_and_split_documents("./data/", None)
    
    conversation_history = []
    while True:
        question = input("请输入问题：")
        
        if question == "exit":
            break
        
        full_answer = ""
        conversation_history.append((question, ""))
        
        async for chunk in pipeline.generate_answer(question, conversation_history, model_name=OllamaModelName):
            print(chunk, end="", flush=True)
            full_answer += chunk
        
        conversation_history[-1] = (question, full_answer)


if __name__ == '__main__':
    asyncio.run(main())
    # asyncio.run(chat_with_history())
