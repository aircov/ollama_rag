# -*- coding: utf-8 -*-
# @Time    : 2025/3/25 10:48
# @Author  : yaomw
# @Desc    :

from extensions import logger


class LLMRerank:
    def __init__(self, llm):
        self.llm = llm
    
    def rerank_documents_with_llm(self, question, docs, top_k):
        """
        使用LLM对检索结果进行重排序
        :param question:
        :param docs:
        :param top_k:
        :return:
        """
        try:
            content_list = [doc.page_content for doc in docs]
            prompt = f"""给定以下查询和文档片段，评估它们的相关性。
            文档是一个列表有多个片段，你需要分析每一个片段，然后进行打分，返回每一个片段的分数，一一对应，以列表格式返回。
            请不要解释你的评分，只返回一个列表，列表长度与文档片段数量相同，列表中的每个元素都是0-10之间的整数。
            评分标准：0分表示完全不相关，10分表示高度相关。
            只需返回0-10之间的整数分数，不要有任何其他解释。

            查询: {question}

            文档片段: {content_list}

            文档片段列表长度: {len(content_list)}

            相关性分数(0-10):"""
            logger.info(f"LLM reranking ...")
            resp = self.llm.invoke(prompt)
            
            llm_score = resp.split("\n\n")[-1].replace("'", "")
            logger.info(f"LLM 评分：{llm_score}")
            llm_score = eval(llm_score)
            if len(llm_score) != len(content_list):
                logger.error(f"LLM 评分长度不匹配：{len(llm_score)} != {len(content_list)}")
                return docs[:top_k]
            else:
                logger.info(f"LLM 评分数量匹配：llm_score：{len(llm_score)} == content_list：{len(content_list)}")
            
            # 按照大模型打分重新排序
            for idx, score in enumerate(llm_score):
                docs[idx].metadata["rerank_score"] = score
            
            ret = sorted(docs, key=lambda x: x.metadata.get("rerank_score"), reverse=True)
            
            ret = [doc for doc in ret if doc.metadata.get("rerank_score", 0) >= 3][:top_k]
            logger.info(f"剔除LLM评分小于3的文档，剩余文档数量：{len(ret)}")
            return ret
        
        except Exception as e:
            logger.error(f"llm rerank Error: {e}")
            return docs[:top_k]
