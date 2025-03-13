# -*- coding: utf-8 -*-
# @Time    : 2025/3/13 15:05
# @Author  : yaomw
# @Desc    :

import gradio as gr

from rag import RAGPipeline
from utils.ollama_utils import fetch_ollama_models

rag = RAGPipeline()


# 新增函数：获取系统使用的模型信息
def get_system_models_info():
    """返回系统使用的各种模型信息"""
    models_info = {
        "嵌入模型": "all-MiniLM-L6-v2",
        "分块方法": "RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)",
        "检索方法": "向量检索 + BM25混合检索 (α=0.7)",
        "重排序模型": "交叉编码器 (sentence-transformers/distiluse-base-multilingual-cased-v2)",
        "生成模型": "deepseek-r1 (7B/1.5B)",
        "分词工具": "jieba (中文分词)"
    }
    return models_info


def process_upload_files(files):
    if not files:
        return "请选择要上传的文件", []
    file_list = rag.load_and_split_documents(files=files)
    
    summary = f"\n总计处理 {len(files)} 个文件，处理完成"
    
    # 已处理文件
    ret = [f"文件名: 《{i['key']}》，分块数量：{i['doc_count']}" for i in file_list]
    
    return summary, "\n".join(ret)

def process_chat(question, history, enable_web_search, model_choice):
    if history is None:
        history = []
    if not question:
        return "请输入问题", history
    
    # 添加用户问题到历史
    history.append((question, ""))
    
    rag.setup_rag_chain(rerank_method="corom", enable_web_search=enable_web_search, model_name=model_choice)


with gr.Blocks() as demo:
    gr.Markdown("# 🧠 智能文档问答系统")
    
    with gr.Tabs() as tabs:
        # 第一个选项卡：问答对话
        with gr.TabItem("💬 问答对话"):
            with gr.Row(equal_height=True):
                # 左侧操作面板 - 调整比例为合适的大小
                with gr.Column(scale=5, elem_classes="left-panel"):
                    gr.Markdown("## 📂 文档处理区")
                    with gr.Group():
                        file_input = gr.File(
                            label="上传文档，支持pdf、docx、txt、md、html",
                            file_types=[],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("🚀 开始处理", variant="primary")
                        upload_status = gr.Textbox(
                            label="处理状态",
                            interactive=False,
                            lines=2
                        )
                        file_list = gr.Textbox(
                            label="已处理文件",
                            interactive=False,
                            lines=3,
                            elem_classes="file-list"
                        )
                    
                    # 将问题输入区移至左侧面板底部
                    gr.Markdown("## ❓ 输入问题")
                    with gr.Group():
                        question_input = gr.Textbox(
                            label="输入问题",
                            lines=3,
                            placeholder="请输入您的问题...",
                            elem_id="question-input"
                        )
                        with gr.Row():
                            # 添加联网开关
                            web_search_checkbox = gr.Checkbox(
                                label="启用联网搜索",
                                value=False,
                                info="打开后将同时搜索网络内容（需要VPN）"
                            )
                            
                            # 添加模型选择下拉框
                            model_choice = gr.Dropdown(
                                choices=fetch_ollama_models(),
                                value="deepseek-r1:14b",
                                label="模型选择",
                                info="选择使用的本地模型",
                                interactive=True
                            )
                        
                        with gr.Row():
                            ask_btn = gr.Button("🔍 开始提问", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes="clear-button",
                                                  scale=1)
                
                # 右侧对话区 - 调整比例
                with gr.Column(scale=7, elem_classes="right-panel"):
                    gr.Markdown("## 📝 对话记录")
                    
                    # 对话记录显示区
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=600,  # 增加高度
                        elem_classes="chat-container",
                        show_label=False
                    )
                    
                    status_display = gr.HTML("", elem_id="status-display")
                    gr.Markdown("""
                                <div class="footer-note">
                                    *回答生成可能需要1-2分钟，请耐心等待<br>
                                    *支持多轮对话，可基于前文继续提问
                                </div>
                                """)
        
        # 第二个选项卡：分块可视化
        with gr.TabItem("📊 分块可视化"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 💡 系统模型信息")
                    
                    # 显示系统模型信息卡片
                    models_info = get_system_models_info()
                    with gr.Group(elem_classes="model-card"):
                        gr.Markdown("### 核心模型与技术")
                        
                        for key, value in models_info.items():
                            with gr.Row():
                                gr.Markdown(f"**{key}**:", elem_classes="label")
                                gr.Markdown(f"{value}", elem_classes="value")
                
                with gr.Column(scale=2):
                    gr.Markdown("## 📄 文档分块统计")
                    refresh_chunks_btn = gr.Button("🔄 刷新分块数据", variant="primary")
                    chunks_status = gr.Markdown("点击按钮查看分块统计")
            
            # 分块数据表格和详情
            with gr.Row():
                chunks_data = gr.Dataframe(
                    headers=["来源", "序号", "字符数", "分词数", "内容预览"],
                    elem_classes="chunk-table",
                    interactive=False,
                    wrap=True,
                    row_count=(10, "dynamic")
                )
            
            with gr.Row():
                chunk_detail_text = gr.Textbox(
                    label="分块详情",
                    placeholder="点击表格中的行查看完整内容...",
                    lines=8,
                    elem_classes="chunk-detail-box"
                )
    
    # 绑定UI事件
    upload_btn.click(
        process_upload_files,
        inputs=[file_input],
        outputs=[upload_status, file_list],
        show_progress=True
    )
    
    # 绑定提问按钮
    ask_btn.click(
        process_chat,
        inputs=[question_input, chatbot, web_search_checkbox, model_choice],
        outputs=[chatbot, question_input]
    )

if __name__ == "__main__":
    demo.launch()
