
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import tempfile

# PDF Loader
from langchain_community.document_loaders import PyPDFLoader

# 문서 분할기
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector DB
from langchain_chroma import Chroma

# OpenAI Embedding / Chat 모델
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


# 최신 Retrieval Chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate

st.title("PDF File Reader")
st.write("---")

# PDF 업로드 영역
uploaded_file = st.file_uploader( "PDF 파일을 업로드하세요",  type=["pdf", "doc", "docx"] )
st.write("---")


# ==========================================================
# PDF → Document 변환 함수
# ==========================================================
def pdf_to_documents(uploaded_file):
    """
    업로드된 PDF 파일을 LangChain Document 형태로 변환

    처리 과정:
    1. Streamlit 업로드 파일 저장
    2. PyPDFLoader 로 PDF 읽기
    3. 페이지 단위 Document 생성

    return:
        pages (list)
    """

    # 임시 폴더 생성 , in메모리 방식으로 저장
    temp_dir = tempfile.TemporaryDirectory()

    # 임시 PDF 파일 경로 생성
    temp_path = os.path.join( temp_dir.name,   uploaded_file.name )

    # 업로드된 파일 저장
    with open(temp_path, "wb") as f: 
        f.write(uploaded_file.getvalue())

    # PDF Loader 생성
    loader = PyPDFLoader(temp_path)

    # PDF 페이지 읽기
    pages = loader.load()
    return pages


if uploaded_file is not None:
    pages = pdf_to_documents(uploaded_file)
    st.success(  f"PDF 로딩 완료 : {len(pages)} 페이지"   )

    text_splitter = RecursiveCharacterTextSplitter(
        # 한 조각의 최대 글자 수
        chunk_size=1000,

        # 앞뒤 중복 문자, 문맥 유지를 위해 사용
        chunk_overlap=200
    )


    # Document 분할
    texts = text_splitter.split_documents(  pages   )

    st.info(  f"문서 조각 개수 : {len(texts)}"   )

    # -------------------------------
    # 3. Embedding 생성
    # -------------------------------
    embeddings = OpenAIEmbeddings()
    # -------------------------------
    # 4. Vector Database 생성
    # -------------------------------
    db = Chroma.from_documents(  documents=texts,  embedding=embeddings   )

    # -------------------------------
    # 5. Retriever 생성
    # -------------------------------
    retriever = db.as_retriever(  search_kwargs={   "k": 3   }   )

    # ======================================================
    # 질문 입력
    # ======================================================
    st.header( "PDF에게 질문하세요"    )
    question = st.text_input( "질문 입력"   , placeholder="예시) 4분기 매출전망을 알려줘.  ")

    if st.button("질문하기"):
        if question.strip()=="":
            st.warning(  "질문을 입력해주세요"     )

        else:
            with st.spinner( "AI가 답변 생성중..."   ):

                llm = ChatOpenAI(  model="gpt-4.1-mini",    temperature=0  )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 문서 분석 전문가입니다.

                    아래 Context 내용을 참고하여
                    질문에 답변하세요.

                    Context: {context}
                    Question: {input}
                    """
                )

                document_chain = (  create_stuff_documents_chain( llm,  prompt   )   )

                qa_chain = create_retrieval_chain(  retriever,  document_chain  )

                result = qa_chain.invoke(  {  "input": question  }  )

                st.write(  result["answer"]    )