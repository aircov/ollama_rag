# -*- coding: utf-8 -*-
# @Time    : 2025/2/26 15:03
# @Author  : yaomw
# @Desc    :

dsl = {
    "create_index": {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,  # 单节点环境，无需复制，否则集群状态是黄色
            "index": {
                "max_ngram_diff": 5
            },
            "analysis": {
                "analyzer": {
                    "ik_smart_synonym": {
                        "type": "custom",
                        "tokenizer": "ik_smart",
                        "filter": [
                            "synonym_filter",
                            "lowercase"
                        ]
                    },
                    "ik_max_word_synonym": {
                        "type": "custom",
                        "filter": [
                            "synonym_filter",
                            "lowercase"
                        ],
                        "tokenizer": "ik_max_word"
                    },
                    "ngram_analyzer": {
                        "tokenizer": "ngram_tokenizer",
                        "filter": [
                            "lowercase"
                        ]
                    }
                },
                "tokenizer": {
                    "ngram_tokenizer": {
                        "type": "ngram",
                        "min_gram": 1,
                        "max_gram": 5,
                        "token_chars": [
                            "letter",
                            "digit"
                        ]
                    }
                },
                "filter": {
                    "synonym_filter": {
                        "ignore_case": "true",
                        "expand": "true",
                        "type": "synonym",
                        "synonyms_path": "analysis/synonym.txt"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word_synonym",
                    "search_analyzer": "ik_smart_synonym",
                    "fields": {
                        "ngram": {
                            "type": "text",
                            "store": False,
                            "analyzer": "ngram_analyzer"
                        }
                    }
                },
                "metadata": {
                    "type": "object"
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {  # HNSW 参数配置（关键修正点！）
                        "type": "hnsw",
                        "m": 16,  # 每个节点的最大连接数
                        "ef_construction": 128  # 索引构建时的候选队列大小
                    }
                }
            }
        }
    }
}
