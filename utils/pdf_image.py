# -*- coding:utf-8 -*-
# @Time      : 2025/8/18 13:31
# @Author    : yaomw
import os

import fitz  # PyMuPDF，用于 PDF 处理

pdf_path = r"../data/集团介绍.pdf"
output_dir = r"./extracted_images"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开PDF文件
pdf_document = fitz.open(pdf_path)
image_paths = []

# 遍历每一页
for page_num in range(len(pdf_document)):
    page = pdf_document[page_num]
    image_list = page.get_images(full=True)  # 获取页面中的所有图像

    # 遍历页面中的每张图像
    for img_index, img in enumerate(image_list):
        xref = img[0]  # 图像的xref引用
        base_image = pdf_document.extract_image(xref)  # 提取图像数据
        image_bytes = base_image["image"]  # 图像的字节数据
        image_ext = base_image["ext"]  # 图像扩展名（如jpeg, png）

        # 保存图像
        image_filename = f"{output_dir}/image_page{page_num + 1}_{img_index}.{image_ext}"
        with open(image_filename, "wb") as image_file:
            image_file.write(image_bytes)
        image_paths.append(image_filename)
        print(f"已保存图像：{image_filename}")
pdf_document.close()
