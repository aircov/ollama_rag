# -*- coding: utf-8 -*-
# @Time    : 2025/3/10 15:16
# @Author  : yaomw
# @Desc    :
import numpy as np
from langchain.embeddings.base import Embeddings
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from typing import List


class ModelScopeEmbeddings(Embeddings):
    def __init__(self, model_name="damo/nlp_corom_sentence-embedding_chinese-base", device='gpu'):
        self.pipeline = pipeline(
            task=Tasks.sentence_embedding,
            model=model_name,
            device=device
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """为文档生成嵌入"""
        inputs = {
            'source_sentence': texts
        }
        results = self.pipeline(inputs)
        # 处理不同模型输出的格式差异
        if isinstance(results, dict):
            embeddings = results['text_embedding'].tolist()
        else:
            embeddings = [result['text_embedding'].tolist() for result in results]
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """为查询生成嵌入"""
        return self.embed_documents([text])[0]
    
    def test_embedding_variance(self):
        test_texts = ["这是第一段", "这是完全不同的第二段内容"]
        embeddings = self.embed_documents(test_texts)
        similarity = np.dot(embeddings[0], embeddings[1])
        print(f"测试文本相似度：{similarity:.4f}")


if __name__ == '__main__':
    embeddings = ModelScopeEmbeddings()
    print(embeddings.embed_documents(["你好"]))
    
    embeddings.test_embedding_variance()
