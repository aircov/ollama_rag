# -*- coding: utf-8 -*-
# @Time    : 2025/3/13 15:55
# @Author  : yaomw
# @Desc    :


async def get_file_list(es, index_name):
    dsl = {
        "size": 0,
        "aggs": {
            "unique_sources": {
                "terms": {
                    "field": "metadata.source.keyword",
                    "size": 1000
                }
            }
        }
    }
    
    resp = await es.search(index=index_name, body=dsl)
    sources = [bucket for bucket in resp['aggregations']['unique_sources']['buckets']]

    return sources