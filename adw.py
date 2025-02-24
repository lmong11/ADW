import os
import ollama
import faiss
import fitz  # PyMuPDF
import pytesseract
import numpy as np
from PIL import Image
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
import streamlit as st
import time
vector_db_path = "faiss_index"

# ------------------- 1. 文档解析 & OCR ------------------- #
def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text")
    return text

# ------------------- 2. 生成向量数据库 ------------------- #
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_texts(["dummy text"], embeddings)

def index_text(text, db_path="faiss_index"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    vector_db.add_texts(docs)
    vector_db.save_local(db_path)  # 这里改为传入参数
    return "索引已更新！"

# ------------------- 3. RAG 检索增强生成 ------------------- #
llm = Ollama(model="deepseek-r1:14b")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever(), chain_type="stuff")

def query_rag(query):
    return qa_chain.run(query)

# ------------------- 4. Agent 智能任务 ------------------- #
def generate_summary(query):
    return f"摘要: 这是关于 '{query}' 的简要总结。"

def generate_visualization(query):
    return f"生成与 '{query}' 相关的数据可视化..."

def generate_translation(query):
    return f"将 '{query}' 翻译成多种语言: 英文, 西班牙语, 法语..."

def generate_code_snippet(query):
    return f"基于 '{query}' 生成代码示例: \nprint('Hello, {query}')"

tools = [
    Tool(
        name="Document Search",
        func=lambda q: vector_db.similarity_search(q, k=5),
        description="查询文档相关内容"
    ),
    Tool(
        name="Report Analysis",
        func=lambda q: f"生成分析报告: {q}",
        description="对文档内容生成结构化分析报告"
    ),
    Tool(
        name="Image Generation",
        func=lambda q: f"生成与 {q} 相关的图片...",
        description="基于查询内容生成相关图片"
    ),
    Tool(
        name="Summary Generation",
        func=generate_summary,
        description="生成文档摘要"
    ),
    Tool(
        name="Data Visualization",
        func=generate_visualization,
        description="基于数据生成可视化图表"
    ),
    Tool(
        name="Translation",
        func=generate_translation,
        description="将查询内容翻译成多种语言"
    ),
    Tool(
        name="Code Generation",
        func=generate_code_snippet,
        description="基于查询内容生成代码示例"
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)

def agent_query(query):
    return agent.run(query)

# ------------------- 5. Streamlit 界面 ------------------- #
st.set_page_config(page_title="智能文档工作流 (ADW)", layout="wide")
st.title("📄 智能文档工作流 (ADW)")

# 文件上传
uploaded_file = st.file_uploader("📂 上传 PDF 文档", type=["pdf"])
if uploaded_file:
    with st.spinner("📄 解析文档中..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        text = extract_text_from_pdf("temp.pdf")
        st.success("✅ 文档解析成功！开始索引...")
        time.sleep(1)
        st.write(index_text(text, vector_db_path))

# 查询输入框
query = st.text_input("🔍 请输入你的查询内容：")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📑 检索文档"):
        with st.spinner("🔍 正在检索..."):
            response = query_rag(query)
        st.success("✅ 检索完成！")
        st.write(response)

with col2:
    if st.button("🤖 使用智能 Agent 查询"):
        with st.spinner("🤖 处理中..."):
            response = agent_query(query)
        st.success("✅ 查询完成！")
        st.write(response)

with col3:
    if st.button("📊 生成分析报告"):
        with st.spinner("📊 生成中..."):
            response = agent.run("生成分析报告: " + query)
        st.success("✅ 报告生成完成！")
        st.write(response)

with col4:
    if st.button("📜 生成摘要"):
        with st.spinner("📜 生成中..."):
            response = generate_summary(query)
        st.success("✅ 摘要生成完成！")
        st.write(response)

st.markdown("---")
st.caption("📌 由 DeepSeek R1 和 LangChain 提供支持")
