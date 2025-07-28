# -*- coding: utf-8 -*-
# @Time    : 2025/2/20 10:40
# @Author  : yaomw
# @Desc    :

import numpy as np

from ollama import Client

from config import OllamaModelName

client = Client(
    host='http://127.0.0.1:11434',
    headers={'Content-Type': "application/json", "Authorization": "Bearer ollama"}
)
models = client.list()
print(models)
print([model.model for model in models.models])

# embed
resp = client.embed(
    model=OllamaModelName,
    input=['深度学习的基本原理', '神经网络的核心概念'],
)
print(resp.embeddings[0][:10])
print("ollama deepseek embedding size:",len(resp.embeddings[0]))

# 向量
vec1 = np.array(resp.embeddings[0])
vec2 = np.array(resp.embeddings[1])

# 计算余弦相似度
cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print(f"Cosine Similarity: {cos_sim:.4f}")

# 点积相似度（适用于已归一化的向量）
dot_product = np.dot(vec1, vec2)
print(f"dot product Similarity: {dot_product}")


stream = client.chat(
    model=OllamaModelName,
    messages=[
        {"role": "system", "content": """You are a helpful assistant. 请用中文回答下面问题"""},
        {'role': 'user', 'content': '解方程 (x²-5x+6=0)。'},
    
    ],
    stream=True,
)
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
    
    
