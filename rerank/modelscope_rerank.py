# -*- coding: utf-8 -*-
# @Time    : 2025/2/28 17:32
# @Author  : yaomw
# @Desc    : CoROM语义相关性-中文
# https://modelscope.cn/models/iic/nlp_corom_passage-ranking_chinese-base-ecom
# https://modelscope.cn/models/iic/nlp_rom_passage-ranking_chinese-base


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 语义相关性排序
model_id_rerank = 'damo/nlp_rom_passage-ranking_chinese-base'
pipeline_se_rerank = pipeline(
    Tasks.text_ranking,
    model=model_id_rerank,
    device='cpu'
)


def rerank(query, docs):
    inputs = {
        'source_sentence': [query],
        'sentences_to_compare': docs
    }
    
    result = pipeline_se_rerank(input=inputs)
    # print(result)  # {'scores': [0.9794721007347107, 0.4872913658618927, 0.029461242258548737]}
    
    # 提取分数和原始数据
    scores = result['scores']
    
    # 将分数和句子打包并排序（按分数从高到低）
    sorted_pairs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    
    # 转换为列表内嵌套字典的格式
    sorted_list = [
        {"text": text, "score": float(score)}
        for score, text in sorted_pairs
    ]
    
    print(sorted_list)
    return sorted_list


if __name__ == '__main__':
    query = "阔腿裤女冬牛仔"
    docs = [
        "阔腿牛仔裤女秋冬款潮流百搭宽松。",
        "牛仔阔腿裤女大码胖mm高腰显瘦夏季薄款宽松垂感泫雅拖地裤子。",
        "阔腿裤男大码高腰宽松。",
    ]
    rerank(query, docs)
