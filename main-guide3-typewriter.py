import os
import tempfile
# LangChain 체인을 별도 작업 스레드에서 실행합니다.
import threading
# 글자별 출력 속도를 조절합니다.
import time
# 작업 스레드에서 생성한 글자를 메인 스레드로 안전하게 전달합니다.
from queue import Empty, Queue

import streamlit as st
from langchain_community.document_loaders import (  PyPDFLoader )
from langchain_text_splitters import (    RecursiveCharacterTextSplitter )
#from langchain_chroma import (    Chroma  )
# Streamlit Cloud 호환성을 위해 Chroma 대신 메모리 저장소를 사용합니다.
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import (  OpenAIEmbeddings,    ChatOpenAI )
from langchain_classic.chains import (   create_retrieval_chain )
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import (   ChatPromptTemplate )
from langchain_core.callbacks import (   BaseCallbackHandler ) #한줄한줄 텍스트 불러오는 라이브러리

st.title("📄 PDF File Reader")
st.write("----------------")


openai_key = st.text_input(  "OPENAI_API_KEY",    type="password" )

uploaded_file = st.file_uploader(   "PDF 파일을 올려주세요",   type=["pdf"] )
st.write("----------------")

def pdf_to_document(uploaded_file):
    """    Streamlit 업로드 PDF를
    LangChain Document 형태로 변환
    """
    # 임시 폴더 생성
    temp_dir = tempfile.TemporaryDirectory()

    # 임시 PDF 파일
    temp_filepath = os.path.join(     temp_dir.name,    uploaded_file.name    )

    with open(   temp_filepath,    "wb"  ) as f:
        f.write(    uploaded_file.getvalue()    )

    loader = PyPDFLoader(   temp_filepath   )

    pages = loader.load()
    return pages

class StreamHandler(BaseCallbackHandler):

    """
    GPT가 토큰을 생성할 때마다
    Streamlit 화면에 출력하는 Handler

    예:
    GPT:   안녕하세요
    생성 과정:
    안
    안녕
    안녕하세요

    처럼 실시간 출력
    """
    def __init__(self, token_queue):
        # 콜백 스레드에서는 UI를 갱신하지 않고 Queue에 글자만 저장합니다.
        self.token_queue = token_queue

    def on_llm_new_token(self, token, **kwargs):
        # UI는 호출하지 않고 생성된 토큰을 한 글자씩 메인 스레드로 전달합니다.
        for character in token:
            self.token_queue.put(character)


def stream_chain_response(qa_chain, question, handler, container):
    """체인은 작업 스레드에서 실행하고 화면은 메인 스레드에서 갱신합니다."""
    result = {}
    error = {}

    # RAG 체인을 실행하고 결과 또는 오류를 공유 변수에 저장합니다.
    def run_chain():
        try:
            result["response"] = qa_chain.invoke(
                {"input": question},
                config={"callbacks": [handler]},
            )
        except Exception as exception:
            error["exception"] = exception

    worker = threading.Thread(target=run_chain, daemon=True)
    worker.start()

    # Streamlit 메인 스레드에서 Queue를 읽어 화면을 한 글자씩 갱신합니다.
    streamed_text = ""
    while worker.is_alive() or not handler.token_queue.empty():
        try:
            token = handler.token_queue.get(timeout=0.05)
            streamed_text += token
            container.markdown(streamed_text)
            time.sleep(0.01)
        except Empty:
            continue

    worker.join()

    if "exception" in error:
        raise error["exception"]

    # 모델이 토큰 콜백을 보내지 않은 경우 최종 답변을 출력합니다.
    if not streamed_text:
        streamed_text = result["response"]["answer"]
        container.markdown(streamed_text)

    return result["response"]

if uploaded_file is not None:
    pages = pdf_to_document(   uploaded_file   )
    st.success(   f"PDF 페이지 : {len(pages)}"  )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(    pages   )

    st.info(  f"문서 조각 : {len(texts)}"  )

    embeddings = OpenAIEmbeddings(  api_key=openai_key   )
    
    # Chroma 변경 -> InMemoryVectorStore
    # db = Chroma.from_documents(
    #     documents=texts,
    #     embedding=embeddings
    # )

    # 분할된 PDF 문서를 메모리 벡터 저장소에 등록합니다.
    db = InMemoryVectorStore(embeddings)
    db.add_documents(texts)

    retriever = db.as_retriever(
        search_kwargs={
            "k":3
        }
    )

    st.header(   "PDF에게 질문하세요"   )
    question = st.text_input(   "질문 입력"    )

    if st.button(   "질문하기"   ):
        if question == "":
            st.warning( "질문을 입력하세요"   )
        else:
            with st.spinner(  "답변 생성중..."     ):

                chat_box = st.empty()
                # StreamHandler와 화면 출력 코드 사이에서 글자를 전달합니다.
                token_queue = Queue()
                handler = StreamHandler(token_queue)

                llm = ChatOpenAI(
                    model="gpt-4.1-mini",
                    temperature=0,
                    api_key=openai_key,
                    streaming=True
                )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 분석 AI 입니다.
                    Context:   {context}
                    Question:  {input}
                    답변:
                    """
                )

                document_chain = ( create_stuff_documents_chain(   llm,    prompt    )   )

                qa_chain = create_retrieval_chain(
                    retriever,
                    document_chain
                )

                # UI 호출을 메인 스레드에서 처리하여 NoSessionContext 오류를 방지합니다.
                stream_chain_response(
                    qa_chain,
                    question,
                    handler,
                    chat_box,
                )
