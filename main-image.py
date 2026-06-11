import base64
import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


load_dotenv()

st.set_page_config(page_title="이미지 질문-답변")
st.title("이미지 질문-답변")
st.caption("이미지를 업로드한 뒤 이미지 내용에 관해 질문해 보세요.")


# 로컬 환경 변수 또는 Streamlit Secrets에서 API 키를 불러옵니다.
def configure_api_key():
    if os.getenv("OPENAI_API_KEY"):
        return True

    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        api_key = None

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return True

    return False


# 이미지 파일을 OpenAI 모델에 전달할 Base64 데이터 URL로 변환합니다.
def create_image_data_url(file_bytes, mime_type):
    encoded_image = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


# 이전 대화와 현재 이미지·질문을 모델 메시지 형식으로 구성합니다.
def create_model_messages(question, image_data_url):
    messages = [
        SystemMessage(
            content=(
                "너는 이미지 분석 전문가다. 업로드된 이미지에서 확인되는 내용을 "
                "근거로 한국어로 정확하게 답하라. 이미지에서 확인할 수 없는 내용은 "
                "추측하지 말고 확인할 수 없다고 답하라."
            )
        )
    ]

    for message in st.session_state.messages:
        if message["role"] == "user":
            messages.append(HumanMessage(content=message["content"]))
        else:
            messages.append(AIMessage(content=message["content"]))

    messages.append(
        HumanMessage(
            content=[
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                },
            ]
        )
    )
    return messages


if not configure_api_key():
    st.error("`.env` 또는 Streamlit Secrets에 `OPENAI_API_KEY`를 설정해 주세요.")
    st.stop()

uploaded_file = st.file_uploader(
    "이미지 파일 선택",
    type=["png", "jpg", "jpeg", "webp"],
    help="PNG, JPG, JPEG, WEBP 이미지를 지원합니다.",
)

if uploaded_file is None:
    st.info("질문을 시작하려면 이미지 파일을 업로드해 주세요.")
    st.stop()

# 업로드된 이미지의 원본 데이터와 고유 식별값을 생성합니다.
image_bytes = uploaded_file.getvalue()
image_id = hashlib.sha256(image_bytes).hexdigest()

# 새로운 이미지가 업로드되면 이전 이미지의 대화 기록을 초기화합니다.
if st.session_state.get("image_id") != image_id:
    st.session_state.image_id = image_id
    st.session_state.messages = []

mime_type = uploaded_file.type or "image/jpeg"
image_data_url = create_image_data_url(image_bytes, mime_type)

# 업로드된 이미지를 화면에 미리 표시합니다.
st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)

# 현재 이미지에 대한 이전 질문과 답변을 다시 표시합니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("이미지에 관해 질문하세요.")
if question:
    # 사용자의 질문과 이미지를 모델에 함께 전달합니다.
    model_messages = create_model_messages(question, image_data_url)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("이미지를 분석하고 있습니다..."):
                llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
                response = llm.invoke(model_messages)

            answer = response.content
            st.write(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except Exception as error:
            st.error(f"답변 생성 중 오류가 발생했습니다: {error}")
