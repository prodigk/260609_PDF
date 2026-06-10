import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

st.set_page_config(page_title="PDF 질문-답변")
st.title("PDF 질문-답변 시스템")
st.caption("PDF를 업로드한 뒤 문서 내용에 관해 질문해 보세요.")


@st.cache_resource(show_spinner=False)
def create_rag_chain(file_bytes: bytes, file_name: str):
    """업로드된 PDF를 벡터화하고 문서 검색 기반 QA 체인을 만든다."""
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        pages = PyPDFLoader(temp_path).load()
        if not pages:
            raise ValueError("PDF에서 읽을 수 있는 페이지를 찾지 못했습니다.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120,
            length_function=len,
            is_separator_regex=False,
        )
        documents = text_splitter.split_documents(pages)
        if not documents:
            raise ValueError("PDF에서 추출할 수 있는 텍스트가 없습니다.")

        for document in documents:
            document.metadata["source"] = file_name

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents)

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            llm=llm,
        )

        system_prompt = (
            "너는 업로드된 PDF의 내용을 바탕으로 답하는 질문-답변 비서다. "
            "아래 context에 포함된 정보만 사용해 한국어로 정확하고 간결하게 답하라. "
            "문서에서 답을 찾을 수 없다면 '업로드된 문서에서 답을 찾을 수 없습니다.'라고 답하라. "
            "추측하거나 문서에 없는 사실을 만들지 마라.\n\n"
            "context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def display_sources(context):
    """검색에 사용된 PDF 페이지를 중복 없이 표시한다."""
    pages = sorted(
        {
            document.metadata["page"] + 1
            for document in context
            if isinstance(document.metadata.get("page"), int)
        }
    )
    if pages:
        st.caption("참조 페이지: " + ", ".join(map(str, pages)))


def get_source_pages(context):
    return sorted(
        {
            document.metadata["page"] + 1
            for document in context
            if isinstance(document.metadata.get("page"), int)
        }
    )


if not os.getenv("OPENAI_API_KEY"):
    st.error("`.env` 파일에 `OPENAI_API_KEY`를 설정해 주세요.")
    st.stop()

uploaded_file = st.file_uploader(
    "PDF 파일 업로드",
    type=["pdf"],
    accept_multiple_files=False,
    help="텍스트가 포함된 PDF 파일을 업로드해 주세요.",
)

if uploaded_file is None:
    st.info("질문을 시작하려면 PDF 파일을 업로드해 주세요.")
    st.stop()

file_bytes = uploaded_file.getvalue()
file_id = hashlib.sha256(file_bytes).hexdigest()
if st.session_state.get("file_id") != file_id:
    st.session_state.file_id = file_id
    st.session_state.messages = []

try:
    with st.spinner(f"`{uploaded_file.name}` 문서를 분석하고 있습니다..."):
        rag_chain = create_rag_chain(file_bytes, uploaded_file.name)
except Exception as error:
    st.error(f"PDF 처리 중 오류가 발생했습니다: {error}")
    st.stop()

st.success(f"`{uploaded_file.name}` 분석이 완료되었습니다.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("pages"):
            st.caption("참조 페이지: " + ", ".join(map(str, message["pages"])))

question = st.chat_input("PDF 내용에 관해 질문하세요.")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("문서에서 답을 찾고 있습니다..."):
                response = rag_chain.invoke({"input": question})
            answer = response["answer"]
            context = response.get("context", [])
            pages = get_source_pages(context)
            st.write(answer)
            display_sources(context)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "pages": pages}
            )
        except Exception as error:
            st.error(f"답변 생성 중 오류가 발생했습니다: {error}")
