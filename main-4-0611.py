# pip install --upgrade langchain-community langchain-text-splitters langchain-openai pypdf python-dotenv streamlit

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 최신 LangChain 패키지 파편화에 맞춘 올바른 Import 경로
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# docs 폴더에 있는 모든 PDF 경로를 가져옵니다.
pdf_paths = sorted(Path("./docs").glob("*.pdf"))
if not pdf_paths:
    st.error("`docs` 폴더에서 PDF 파일을 찾을 수 없습니다.")
    st.stop()

# 각 PDF의 페이지를 하나의 목록으로 합칩니다.
pages = []
for pdf_path in pdf_paths:
    file_pages = PyPDFLoader(str(pdf_path)).load()
    for page in file_pages:
        # 검색 결과에서 원본 파일을 확인할 수 있도록 파일명을 저장합니다.
        page.metadata["source"] = pdf_path.name
    pages.extend(file_pages)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap  = 20,        # 청크 간 문맥 연결을 위해 겹칠 글자 수
    length_function = len,      # 길이 측정 기준 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호의 정규표현식 해석 여부
)
texts = text_splitter.split_documents(pages)
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
# 모든 PDF 문서를 하나의 메모리 벡터 저장소에 추가합니다.
db = InMemoryVectorStore(embeddings_model)
db.add_documents(texts)

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

# RAG 체인 생성: 검색된 문서들을 활용하여 질문에 답하는 체인과 멀티 쿼리 리트리버를 연결
rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)

# 실제 질문에 대한 답변 생성 --------------------------------------
#question = "아내가 먹고 싶어하는 음식은 무엇이야?"
# question = "26년 2분기 이후 네이버 매출 전망이 어떻게 돼?"
# response = rag_chain.invoke({"input": question})

# 결과 출력
# print(f"검색된 참조 문서 개수: {len(response.get('context', []))}")
# print(f"답변: {response['answer']}")
# print("----------- [최종 답변] -----------")
# print(response['answer'])
#-------------------------------------------------------------


st.title("RAG 기반 질문-답변 시스템")
# 앱에서 미리 로드된 PDF 파일명을 표시합니다.
st.caption(
    "미리 로드된 PDF: " + ", ".join(pdf_path.name for pdf_path in pdf_paths)
)
user_question = st.text_input("네이버 투자 적합성에 대한 질문을 입력하세요:")
if user_question:
    response = rag_chain.invoke({"input": user_question})
    st.write(f"검색된 참조 문서 개수: {len(response.get('context', []))}")
    st.write(f"답변: {response['answer']}")
    
    
