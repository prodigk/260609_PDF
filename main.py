# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

loader = PyPDFLoader("NAVER_20260607.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap  = 20,        # 청크 간 문맥 연결을 위해 겹칠 글자 수
    length_function = len,      # 길이 측정 기준 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호의 정규표현식 해석 여부
)
texts = text_splitter.split_documents(pages)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma.from_documents(texts, embeddings_model)

# 멀티 쿼리 리트리버 생성 및 LLM 연결
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

# 사용자의 질문을 다양한 각도에서 재해석하여 검색 확률을 높이는 MultiQueryRetriever를 생성
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), 
    llm=llm
)

# RAG 체인 생성
# LLM에게 전달할 프롬프트를 정의
system_prompt = (
    "너는 질문-답변을 돕는 유능한 비서야. "
    "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
    "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 검색된 문서들을 활용하여 질문에 답하는 체인 생성
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Chroma DB 
retriever = db.as_retriever()

# RAG 체인 생성: 검색된 문서들을 활용하여 질문에 답하는 체인과 멀티 쿼리 리트리버를 연결
rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)

# 실제 질문에 대한 답변 생성
#question = "아내가 먹고 싶어하는 음식은 무엇이야?"
question = "26년 2분기 이후 네이버 매출 전망이 어떻게 돼?"

response = rag_chain.invoke({"input": question})

# 결과 출력
# print(f"검색된 참조 문서 개수: {len(response.get('context', []))}")
# print(f"답변: {response['answer']}")

print("----------- [최종 답변] -----------")
print(response['answer'])


# Streamlit 앱으로 질문-답변 시스템 구현
import streamlit as st

st.title("RAG 기반 질문-답변 시스템")
user_question = st.text_input("네이버 투자 적합성에 대한 질문을 입력하세요:")
if user_question:
    response = rag_chain.invoke({"input": user_question})
    st.write(f"검색된 참조 문서 개수: {len(response.get('context', []))}")
    st.write(f"답변: {response['answer']}")
    
    
