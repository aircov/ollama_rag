# -*- coding: utf-8 -*-
# @Time    : 2025/3/13 15:48
# @Author  : yaomw
# @Desc    :
from ollama import Client

from config import OllamaUrl

# 初始化Ollama客户端
ollama_client = Client(
    host=OllamaUrl,
    headers={'Content-Type': "application/json", "Authorization": "Bearer ollama"}
)


def fetch_ollama_models():
    models = ollama_client.list()
    return [model.model for model in models.models]
