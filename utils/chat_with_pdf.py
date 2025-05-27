# -*- coding: utf-8 -*-
# @Time    : 2025/4/25
# @Author  : yaomingw
# @Desc    : https://docs.mistral.ai/capabilities/vision/
import os
from mistralai import Mistral

api_key = "QS1L4eQPc8dy4q8kthQXHfjZIqe1UhY0"

# Specify model
model = "mistral-small-latest"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# If local document, upload and retrieve the signed url
# uploaded_pdf = client.files.upload(
#     file={
#         "file_name": "uploaded_file.pdf",
#         "content": open("uploaded_file.pdf", "rb"),
#     },
#     purpose="ocr"
# )
# signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)

# Define the messages for the chat
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "what is the last sentence in the document"
            },
            {
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/1805.04770"
                # "document_url": signed_url.url
            }
        ]
    }
]

chat_response = client.chat.complete(
    model=model,
    messages=messages
)

print(chat_response.choices[0].message.content)

# Output:
# The last sentence in the document is:\n\n\"Zaremba, W., Sutskever, I., and Vinyals, O. Recurrent neural network regularization. arXiv:1409.2329, 2014.

print("###" * 30)


# Specify model
model = "pixtral-12b-2409"

client = Mistral(api_key=api_key)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What's in this image?"
            },
            {
                "type": "image_url",
                "image_url": "https://tripfixers.com/wp-content/uploads/2019/11/eiffel-tower-with-snow.jpeg"
            }
        ]
    }
]

# Get the chat response
chat_response = client.chat.complete(
    model=model,
    messages=messages
)


print(chat_response.choices[0].message.content)

