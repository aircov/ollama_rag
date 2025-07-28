# -*- coding: utf-8 -*-
# @Time    : 2025/6/16
# @Author  : yaomingw
# @Desc    :
from typing import List

from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch


class Qwen3Embeddings(Embeddings):
    def __init__(self, model_name="qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name
        # 加载模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        with torch.no_grad():
            # 待处理的文本
            for text in texts:
                # 对文本进行编码
                inputs = self.tokenizer(text, return_tensors="pt")
                # 获取模型输出
                outputs = self.model(**inputs)
                # 获取模型输出
                embedding = outputs.last_hidden_state[:, -1, :]

                embeddings.append(embedding.tolist()[0])

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, -1, :]
        return embedding.tolist()[0]


if __name__ == '__main__':
    qwen3_embeddings = Qwen3Embeddings()
    print(qwen3_embeddings.embed_query("人工智能正在改变我们的生活"))
