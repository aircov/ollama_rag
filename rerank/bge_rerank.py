# -*- coding: utf-8 -*-
# @Time    : 2025/7/17
# @Author  : yaomingw
# @Desc    : 重排序
# https://mp.weixin.qq.com/s/UxIZwR2w5Hcp-RecsgnckA
# embeding：https://modelscope.cn/models/BAAI/bge-m3
# reranker：http://modelscope.cn/models/BAAI/bge-reranker-v2-m3


from FlagEmbedding import FlagLLMReranker


from FlagEmbedding import FlagAutoModel


# https://github.com/FlagOpen/FlagEmbedding/tree/master/examples/inference/embedder
sentences_1 = ["样例数据-1", "样例数据-2"]
sentences_2 = ["样例数据-3", "样例数据-4"]
model = FlagAutoModel.from_finetuned('BAAI/bge-large-zh-v1.5',
                                     query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                     use_fp16=True,
                                     devices=['cuda:1']) # Setting use_fp16 to True speeds up computation with a slight performance degradation
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)

# for s2p(short query to long passage) retrieval task, suggest to use encode_queries() which will automatically add the instruction to each query
# corpus in retrieval task can still use encode_corpus(), since they don't need instruction
queries = ['query_1', 'query_2']
passages = ["样例文档-1", "样例文档-2"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode_corpus(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)



reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# reranker = FlagLLMReranker('BAAI/bge-reranker-v2-gemma', use_bf16=True) # You can also set use_bf16=True to speed up computation with a slight performance degradation


# 重排序器以问题和文档作为输入
score = reranker.compute_score(['query', 'passage'])
print(score)

scores = reranker.compute_score([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']])
print(scores)
