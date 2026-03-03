import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
st.title("RAG项目测试")

client= OpenAI(api_key=deepseek_api_key,base_url="https://api.deepseek.com/v1")
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "你是一个AI学术助手。"}]
for msg in st.session_state.messages:
    if msg["role"] == "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
if prompt := st.chat_input("请输入你的问题"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        response = client.chat.completions.create(model="deepseek-chat",messages=st.session_state.messages,stream=True)
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})