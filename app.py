from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

# 파일 경로 설정
FILE_PATH = "NAVER_20260607.pdf"
loader = PyPDFLoader(FILE_PATH)

# PDF 로더 초기화
docs = loader.load()

st.title("PDF Document 로더 샘플")

# 문서의 내용 출력
st.write(docs[0].page_content[:200])

