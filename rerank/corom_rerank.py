# -*- coding: utf-8 -*-
# @Time    : 2025/2/28 17:32
# @Author  : yaomw
# @Desc    : CoROM语义相关性-中文
# https://modelscope.cn/models/iic/nlp_corom_passage-ranking_chinese-base-ecom  电商领域
# https://modelscope.cn/models/iic/nlp_rom_passage-ranking_chinese-base  通用领域
from typing import List

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class CoROMRerank:
    def __init__(self, model_name='damo/nlp_rom_passage-ranking_chinese-base', device='gpu'):
        self.pipeline_se_rerank = pipeline(
            Tasks.text_ranking,
            model=model_name,
            device=device
        )
    
    def rerank(self, query: str, docs: List):
        # 语义相关性排序
        
        inputs = {
            'source_sentence': [query],
            'sentences_to_compare': docs
        }
        
        result = self.pipeline_se_rerank(input=inputs)
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
        
        # print(sorted_list)
        return sorted_list


if __name__ == '__main__':
    query = "功和功率的区别"
    docs = [
        "功反映做功多少，功率反映做功快慢。",
        "什么是有功功率和无功功率?无功功率有什么用什么是有功功率和无功功率?无功功率有什么用电力系统中的电源是由发电机产生的三相正弦交流电,在交>流电路中,由电源供给负载的电功率有两种;一种是有功功率,一种是无功功率。",
        "优质解答在物理学中,用电功率表示消耗电能的快慢．电功率用P表示,它的单位是瓦特（Watt）,简称瓦（Wa）符号是W.电流在单位时间内做的功叫做电功率 以灯泡为例,电功率越大,灯泡越亮.灯泡的亮暗由电功率（实际功率）决定,不由通过的电流、电压、电能决定!",
    ]
    rerank = CoROMRerank()
    print(rerank.rerank(query, docs))
