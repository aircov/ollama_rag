# -*- coding: utf-8 -*-
# @Time    : 2025/3/13 15:05
# @Author  : yaomw
# @Desc    :
import time

import gradio as gr

from rag import RAGPipeline
from utils.ollama_utils import fetch_ollama_models
from utils.utils import get_file_list

from config import OllamaModelName

rag = RAGPipeline(OllamaModelName)

# 定义全局停止标志变量
stop_flag = False


def stop_chat():
    global stop_flag
    stop_flag = True
    # 此处可以返回一个状态信息供界面显示，也可以不返回
    return "回答已中断"


def process_upload_files(files):
    if not files:
        # 获取es中的文件列表
        file_list = get_file_list(rag.es_client, rag.es_index_name)
        
        # 已处理文件
        ret = [f"文件名: 《{i['key']}》，分块数量：{i['doc_count']}" for i in file_list]
        return "请选择要上传的文件", "\n".join(ret)
    
    file_list = rag.load_and_split_documents(files=files)
    
    summary = f"\n总计处理 {len(files)} 个文件，处理完成"
    
    # 已处理文件
    ret = [f"文件名: 《{i['key']}》，分块数量：{i['doc_count']}" for i in file_list]
    
    return summary, "\n".join(ret)


def process_chat(question, history, enable_web_search, model_choice):
    # 初始化历史记录
    
    print(
        f"\nrag问答参数:\n question:{question}, history:{history}, 联网搜索:{enable_web_search}, 模型选择:{model_choice}")
    
    global stop_flag
    stop_flag = False  # 重置停止标志，每次生成前确保为 False
    
    history = history or []
    
    if not question.strip():
        print("请输入问题")
        yield history
        return
    
    # 添加用户问题到历史
    history.append((question, ""))
    
    # 立即清空输入框并显示用户问题
    yield history, ""
    
    chain = rag.setup_rag_chain(rerank_method="corom", enable_web_search=enable_web_search)
    
    # 初始化答案
    full_answer = ""
    
    thinking_mode = False  # 是否处于思考过程
    
    think_time = time.time()
    
    # 流式获取回答
    for chunk in chain.stream(question):
        
        print(chunk, end="", flush=True)
        
        if stop_flag:  # 检查是否点击了停止按钮
            full_answer += "\n\n回答已中断"
            history[-1] = (question, full_answer)
            yield history, ""
            return  # 中断后直接结束
        
        full_answer += chunk
        
        # 检测思考过程开始，确保只触发一次
        if "<think>" in full_answer and not thinking_mode:
            thinking_mode = True
            # 用 <details open> 和 <summary>思考中...</summary> 替换 <think>
            full_answer = full_answer.replace(
                "<think>",
                "<details open>\n<summary>思考中...</summary>\n"
            )
        
        # 如果在思考过程中，检查是否到达结束标记
        if thinking_mode and "</think>" in full_answer:
            # 将 </think> 替换为 </details> 并更新 summary 文本
            full_answer = full_answer.replace("</think>",
                                              f"\n</details>\n\n ----------思考耗时:{(time.time() - think_time):.2f}s----------")
            full_answer = full_answer.replace("思考中...", "思考完成")
            thinking_mode = False
        
        # 更新最后一条历史记录
        history[-1] = (question, full_answer)
        # time.sleep(0.02)
        # 逐步返回更新后的对话状态
        yield history, ""
    
    # 如果循环正常结束，也确保返回最终结果
    yield history, ""


def clear_chat_history():
    return None, "对话已清空"


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
                                info="选择使用本地模型或云端模型"
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
                    
                    # 在对话记录区域中添加停止按钮
                    stop_btn = gr.Button("⏹️ 停止回答", variant="secondary")
                    
                    status_display = gr.HTML("", elem_id="status-display")
                    gr.Markdown("""
                    <div class="footer-note">
                        *回答生成可能需要1-2分钟，请耐心等待<br>
                        *支持多轮对话，可基于前文继续提问
                    </div>
                    """)
    
    # 绑定UI事件
    upload_btn.click(
        process_upload_files,
        inputs=[file_input],
        outputs=[upload_status, file_list]
    )
    
    # 绑定提问按钮
    ask_btn.click(
        process_chat,
        inputs=[question_input, chatbot, web_search_checkbox, model_choice],
        outputs=[chatbot, question_input]
    )
    
    # 绑定清空按钮
    clear_btn.click(
        clear_chat_history,
        inputs=[],
        outputs=[chatbot, status_display]
    )
    
    # 绑定停止按钮事件，中断流式回答
    stop_btn.click(
        stop_chat,
        inputs=[],
        outputs=[]
    )

if __name__ == "__main__":
    demo.queue(max_size=5).launch(server_name="0.0.0.0", server_port=7860)