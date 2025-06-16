# -*- coding: utf-8 -*-
# @Time    : 2025/6/16
# @Author  : yaomingw
# @Desc    :


# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0
#
# from sentence_transformers import SentenceTransformer
#
# # Load the model
# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
#
# # The queries and documents to embed
# queries = [
#     "What is the capital of China?",
#     "Explain gravity",
# ]
# documents = [
#     "The capital of China is Beijing.",
#     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
# ]
#
# # Encode the queries and documents. Note that queries benefit from using a prompt
# # Here we use the prompt called "query" stored under `model.prompts`, but you can
# # also pass your own prompt via the `prompt` argument
# query_embeddings = model.encode(queries, prompt_name="query")
# document_embeddings = model.encode(documents)
#
# # Compute the (cosine) similarity between the query and document embeddings
# similarity = model.similarity(query_embeddings, document_embeddings)
# print(similarity)
# # tensor([[0.7646, 0.1414],
# #         [0.1355, 0.6000]])



from transformers import AutoTokenizer, AutoModel
import torch

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_pretrained("qwen/Qwen3-Embedding-0.6B")

# 待处理的文本
text = "人工智能正在改变我们的生活"
# 对文本进行编码
inputs = tokenizer(text, return_tensors="pt")

# 获取模型输出
with torch.no_grad():
    outputs = model(**inputs)
# 提取文本向量
embedding = outputs.last_hidden_state[:, -1, :]
print(embedding)
