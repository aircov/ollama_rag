# -*- coding: utf-8 -*-
# @Time    : 2025/6/16
# @Author  : yaomingw
# @Desc    :

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen3-Reranker-0.6B")
model = AutoModelForSequenceClassification.from_pretrained("qwen/Qwen3-Reranker-0.6B")
# 用户查询
query = "如何提高学习效率"
# 候选文档列表
documents = ["多做练习题可以提高学习成绩", "制定合理的学习计划有助于提高学习效率", "保持良好的心态对学习很重要"]
# 对查询和文档进行编码
input_pairs = []
for doc in documents:
    input_pair = tokenizer(query, doc, return_tensors="pt")
    input_pairs.append(input_pair)
# 获取模型输出
scores = []
for input_pair in input_pairs:
    with torch.no_grad():
        outputs = model(**input_pair)
        logits = outputs.logits
        score = torch.softmax(logits, dim=1)[0][1].item()
        scores.append(score)
# 根据得分对文档进行排序
sorted_documents = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
print(sorted_documents)