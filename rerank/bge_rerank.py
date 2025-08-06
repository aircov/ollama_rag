# -*- coding: utf-8 -*-
# @Time    : 2025/7/17
# @Author  : yaomingw
# @Desc    : 重排序
# https://mp.weixin.qq.com/s/UxIZwR2w5Hcp-RecsgnckA
# https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder


from FlagEmbedding import FlagLLMReranker, FlagAutoModel, FlagAutoReranker
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"正在使用设备: {device}")

sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagAutoModel.from_finetuned(
    'BAAI/bge-large-zh-v1.5',
    # 'BAAI/bge-small-zh-v1.5',
    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
    use_fp16=True,
    devices=[device]
)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
print(embeddings_1.shape)  # large 1024维度， small 512维度
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
queries = ['如何提高学习效率', 'query_2']
passages = ["多做练习题可以提高学习成绩", "制定合理的学习计划有助于提高学习效率", "保持良好的心态对学习很重要"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)


reranker = FlagAutoReranker.from_finetuned(
    'BAAI/bge-reranker-large',
    query_max_length=256,
    passage_max_length=512,
    use_fp16=True,
    devices=[device]
)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

score = reranker.compute_score(['query', 'passage'])
print(score)  # -1.5263671875

# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
score = reranker.compute_score(['query', 'passage'], normalize=True)
print(score)  # 0.1785258315203034

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)  # [-5.60546875, 5.76171875]

# You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']],
                                normalize=True)
print(scores)  # [0.0036642203307843528, 0.9968641641227171]
