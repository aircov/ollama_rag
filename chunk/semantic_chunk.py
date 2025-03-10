# -*- coding: utf-8 -*-
# @Time    : 2025/2/28 17:59
# @Author  : yaomw
# @Desc    : rag语义分块
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document
from typing import List
import re

from embed.modelscope_embeddings import ModelScopeEmbeddings


def load_pdf_content(pdf_path: str) -> str:
    """加载并预处理PDF内容"""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # 合并并清理文本
    full_text = "\n".join([p.page_content for p in pages])
    cleaned_text = re.sub(r'\n{3,}', '\n\n', full_text)
    return cleaned_text.strip()


def semantic_chunk(text: str, embed_model) -> List[Document]:
    """执行语义分块"""
    chunker = SemanticChunker(
        embed_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=0.92  # 根据模型特性调整
    )
    return chunker.create_documents([text])



if __name__ == "__main__":

    embed_model = ModelScopeEmbeddings(
        model_name='damo/nlp_corom_sentence-embedding_chinese-base',  # 推荐专用嵌入模型
        device='cpu'  # 可改为 'cuda:0' 使用 GPU
    )
    
    # 加载并处理PDF
    raw_text = load_pdf_content("../data/集团介绍.pdf")
    print(f"原始文本总长度：{len(raw_text)}字符")
    
    # 执行语义分块
    chunks = semantic_chunk(raw_text, embed_model)
    
    # 输出分块结果
    print(f"\n生成分块数量：{len(chunks)}")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n分块 #{i}（长度：{len(chunk.page_content)}字符）")
        print("-" * 50)
        print(chunk.page_content)
