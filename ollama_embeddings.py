# -*- coding: utf-8 -*-
# @Time    : 2025/2/25 15:17
# @Author  : yaomw
# @Desc    :
from typing import List

import numpy as np
import requests
from langchain_core.embeddings import Embeddings


class OllamaEmbeddings(Embeddings):
    """自定义Ollama嵌入模型"""
    
    def __init__(self, base_url, model):
        self.base_url = base_url
        self.model = model
    
    def _normalize(self, vector: List[float]) -> List[float]:
        """L2归一化处理"""
        norm = np.linalg.norm(vector)
        return (vector / norm).tolist() if norm > 0 else vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 批量生成文档嵌入
        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            response.raise_for_status()
            emb = self._normalize(response.json()["embedding"])
            embeddings.append(emb)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        # 生成查询嵌入
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return self._normalize(response.json()["embedding"])
