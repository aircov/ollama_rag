# -*- coding: utf-8 -*-
# @Time    : 2025/3/11 16:09
# @Author  : yaomw
# @Desc    :

from googlesearch import search


def google_search(query, num_results):
    results = search(query, num_results, advanced=True)
    
    search_results = [
        {
            "title": result.title,
            "url": result.url,
            "description": result.description
        }
        for result in results
    ]
    
    return search_results


if __name__ == '__main__':
    query = "萧山天气"
    results = google_search(query, 20)
    print(results)
