import streamlit as st
from app.streamlit_app import get_bedrock_response
from app.prompt_response import get_response_with_prompt

# if __name__ == "__main__":
#     print(get_response_with_prompt("カレーの作り方を教えてください。"))

st.title("AWS Bedrockのデモアプリ")

user_input = st.text_input("メッセージを入力してください")

if st.button("送信"):
    result = get_bedrock_response(user_input)
    if result:
        st.write(result)
