# -*- coding: utf-8 -*-
# @Time    : 2025/3/11 16:09
# @Author  : yaomw
# @Desc    :

from baidusearch.baidusearch import search


result = search("盗龄医生", 10, debug=1)
print(result)

from googlesearch import search

results = search("盗龄医生", 10)
print([i for i in results])