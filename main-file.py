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
st.title("📄 PDF 질문-답변 시스템")
st.caption("PDF를 업로드한 뒤 문서 내용에 관해 질문해 보세요.")

st.markdown(
    """
    <style>
    :root {
        --composer-width: 46rem;
        --composer-side-gap: 1rem;
    }

    [data-testid="stFileUploader"] {
        position: fixed;
        left: 50%;
        bottom: 6.75rem;
        transform: translateX(-50%);
        width: min(
            var(--composer-width),
            calc(100vw - (var(--composer-side-gap) * 2))
        );
        z-index: 999;
        box-sizing: border-box;
        padding: 0.65rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.9rem;
        background: var(--background-color);
        box-shadow: 0 0.35rem 1.25rem rgba(0, 0, 0, 0.12);
        max-height: 10rem;
        overflow-y: auto;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 3.25rem;
        padding: 0.55rem 0.75rem;
        border-radius: 0.65rem;
        background: rgba(128, 128, 128, 0.06);
    }

    [data-testid="stBottomBlockContainer"] {
        width: min(
            var(--composer-width),
            calc(100vw - (var(--composer-side-gap) * 2))
        );
        margin: 0 auto;
        padding-bottom: 0.75rem;
    }

    [data-testid="stChatInput"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.9rem;
        background: var(--background-color);
        box-shadow: 0 0.35rem 1.25rem rgba(0, 0, 0, 0.12);
    }

    [data-testid="stAppViewBlockContainer"] {
        padding-bottom: 18rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(128, 128, 128, 0.025);
        border-color: rgba(128, 128, 128, 0.2);
        border-radius: 0.9rem;
    }

    @media (max-width: 640px) {
        :root {
            --composer-side-gap: 0.5rem;
        }

        [data-testid="stFileUploader"] {
            bottom: 6.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_pdf(file_name, file_bytes):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        pages = PyPDFLoader(temp_path).load()
        for page in pages:
            page.metadata["source"] = file_name
        return pages
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@st.cache_resource(show_spinner=False)
def create_rag_chain(files: tuple[tuple[str, bytes], ...]):
    """업로드된 모든 PDF를 하나의 문서 컬렉션으로 벡터화한다."""
    pages = []

    for file_name, file_bytes in files:
        pages.extend(load_pdf(file_name, file_bytes))

    if not pages:
        raise ValueError("PDF에서 읽을 수 있는 페이지를 찾지 못했습니다.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )
    documents = text_splitter.split_documents(pages)
    if not documents:
        raise ValueError("PDF에서 추출할 수 있는 텍스트가 없습니다.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    retriever = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        llm=llm,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 업로드된 PDF의 내용을 바탕으로 답하는 질문-답변 비서다. "
                "아래 context에 포함된 정보만 사용해 한국어로 정확하고 간결하게 답하라. "
                "문서에서 답을 찾을 수 없다면 "
                "'업로드된 문서에서 답을 찾을 수 없습니다.'라고 답하라. "
                "추측하거나 문서에 없는 사실을 만들지 마라.\n\n"
                "context:\n{context}",
            ),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


def get_source_references(context):
    """검색에 사용된 파일명과 페이지를 중복 없이 반환한다."""
    return sorted(
        {
            (
                document.metadata.get("source", "알 수 없는 파일"),
                document.metadata["page"] + 1,
            )
            for document in context
            if isinstance(document.metadata.get("page"), int)
        },
        key=lambda reference: (reference[0].lower(), reference[1]),
    )


def display_sources(references):
    if references:
        source_text = " · ".join(
            f"{file_name} p.{page}" for file_name, page in references
        )
        st.caption(f"참조 문서: {source_text}")


def display_message(message):
    with st.chat_message(message["role"]):
        st.write(message["content"])
        display_sources(message.get("references", []))


if not os.getenv("OPENAI_API_KEY"):
    st.error("`.env` 파일에 `OPENAI_API_KEY`를 설정해 주세요.")
    st.stop()

uploaded_files = st.file_uploader(
    "PDF 파일 추가",
    type=["pdf"],
    accept_multiple_files=True,
    help="새 PDF를 추가해도 기존에 선택한 문서는 유지됩니다.",
    label_visibility="collapsed",
)

if not uploaded_files:
    st.info("질문을 시작하려면 PDF 파일을 업로드해 주세요.")
    st.chat_input("먼저 PDF 파일을 업로드해 주세요.", disabled=True)
    st.stop()

files = tuple(
    (uploaded_file.name, uploaded_file.getvalue())
    for uploaded_file in uploaded_files
)

st.session_state.setdefault("messages", [])

try:
    with st.spinner(f"PDF {len(files)}개를 분석하고 있습니다..."):
        rag_chain = create_rag_chain(files)
except Exception as error:
    st.error(f"PDF 처리 중 오류가 발생했습니다: {error}")
    st.stop()

file_names = ", ".join(file_name for file_name, _ in files)
st.success(f"문서 {len(files)}개 분석 완료: {file_names}")

st.caption("대화 기록")
chat_history = st.container(height=420, border=True)

with chat_history:
    if not st.session_state.messages:
        st.info("업로드한 문서에 관해 질문을 입력해 주세요.")

    for message in st.session_state.messages:
        display_message(message)

question = st.chat_input("PDF 내용에 관해 질문하세요.")
if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with chat_history:
        display_message({"role": "user", "content": question})
        with st.chat_message("assistant"):
            try:
                with st.spinner("문서에서 답을 찾고 있습니다..."):
                    response = rag_chain.invoke({"input": question})
                answer = response["answer"]
                context = response.get("context", [])
                references = get_source_references(context)
                st.write(answer)
                display_sources(references)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "references": references,
                    }
                )
            except Exception as error:
                st.error(f"답변 생성 중 오류가 발생했습니다: {error}")
