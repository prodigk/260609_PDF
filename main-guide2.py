import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import (  PyPDFLoader )
from langchain_text_splitters import (    RecursiveCharacterTextSplitter )
from langchain_chroma import (    Chroma  )
from langchain_openai import (  OpenAIEmbeddings,    ChatOpenAI )
from langchain_classic.chains import (   create_retrieval_chain )
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import (   ChatPromptTemplate )

st.title("PDF File Reader")

st.write("----------------")

########################
openai_key = st.text_input(  "OPENAI API KEY 입력",    type="password" )

uploaded_file = st.file_uploader(  "PDF 파일을 올려주세요",   type=["pdf"] )
st.write("----------------")

def pdf_to_document(uploaded_file):
    """
    Streamlit 업로드 PDF 파일을
    LangChain Document 객체로 변환

    처리:
    PDF 저장
    ↓
    PyPDFLoader
    ↓
    Document 생성

    """
    temp_dir = tempfile.TemporaryDirectory()

    temp_filepath = os.path.join(    temp_dir.name,    uploaded_file.name   )

    with open(   temp_filepath,    "wb"    ) as f:
        f.write(  uploaded_file.getvalue()   )

    loader = PyPDFLoader(  temp_filepath   )
    pages = loader.load()
    return pages



if uploaded_file is not None:
    pages = pdf_to_document(    uploaded_file   )
    st.success(   f"PDF 페이지 수 : {len(pages)}"    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    texts = text_splitter.split_documents(    pages   )
    st.info( f"분할 문서 개수 : {len(texts)}"  )

    # ------------------------------------------------------
    # 3. Embedding 생성
    # 텍스트 -> 벡터 변환
    # 의미 검색 가능
    # ------------------------------------------------------
    embeddings_model = OpenAIEmbeddings(  api_key=openai_key    )

    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings_model
    )

    # ------------------------------------------------------
    # 5. Retriever 생성
    # 질문과 비슷한 PDF 조각 검색
    # ------------------------------------------------------
    # 벡터 데이터베이스(Vector Store)를 데이터를 찾아오는 '검색기(Retriever)'로 변환하는 역할
    retriever = db.as_retriever( search_kwargs={ "k":3  }  ) # 가장 유사도가 높은(가장 관련이 깊은) 문서 딱 3개만

    st.header(  "PDF에게 질문하세요"   )
    question = st.text_input(   "질문 입력"    )

    if st.button(   "질문하기" ):
        if question == "":
            st.warning(  "질문을 입력 안했네~~"   )
        else:
            with st.spinner(   "GPT 답변 생성중..."    ):
                llm = ChatOpenAI(  model="gpt-4.1-mini", temperature=0,  api_key=openai_key    )

                prompt = ChatPromptTemplate.from_template(
                    """
                    당신은 PDF 문서 분석 AI입니다.
                    아래 Context 내용을 이용해서
                    질문에 답변하세요.
                    Context:  {context}
                    질문: {input}
                    답변:
                    """
                )

                document_chain = (  create_stuff_documents_chain(  llm,  prompt   )  )
                qa_chain = create_retrieval_chain(   retriever,  document_chain   )
                response = qa_chain.invoke(   {  "input": question   }    )
                st.write(  response["answer"]    )
