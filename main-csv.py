import csv
import hashlib
import io
import threading
import time
from queue import Empty, Queue

import streamlit as st
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


FIXED_QUESTIONS = [
    (
        "2024년 3월 서울 종로구 명륜2가에 있는 아남1 아파트의 "
        "건축년도와 도로명 주소 정보를 알려주세요."
    ),
    (
        "명륜2가 아남1 아파트의 거래 내역 중에서, 거래금액이 "
        "125,000만 원 이상이거나 3월 25일 이후에 계약된 건의 "
        "층수는 각각 몇 층인가요?"
    ),
    (
        "2024년 3월에 계약된 명륜2가 아남1 아파트(전용면적 84.9㎡)의 "
        "최고 거래금액과 최저 거래금액은 각각 얼마이며, 두 거래의 "
        "계약일은 며칠인지 비교하여 요약해 주세요."
    ),
]




st.set_page_config(page_title="부동산 데이터 CSV 질문-답변")
st.title("부동산 데이터 CSV 질문-답변")

st.markdown(
    """
    <div style="
        display: flex;
        align-items: center;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-bottom: 0.75rem;
    ">
        <span style="
            display: inline-flex;
            align-items: center;
            padding: 0.15rem 0.5rem;
            border-radius: 999px;
            background: rgba(33, 195, 84, 0.15);
            color: rgb(20, 135, 60);
            font-size: 0.75rem;
            font-weight: 700;
            line-height: 1.4;
        ">
            GUIDE
        </span>
        <small style="line-height: 1.4;">
            OPENAI API KEY 입력과 CSV를 업로드한 뒤 고정 질문을 선택하세요.
            <a
                href="https://blog.open-network.co.kr/openai-api-guide"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="OPENAI API KEY 발급 방법 (새 창에서 열림)"
            >
                OPENAI API KEY 발급 방법 ↗
            </a>
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)

# 고정 질문 버튼은 일반 텍스트 크기에서 굵게 표시합니다.
st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button p {
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.4;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

openai_key = st.text_input(
    "OPENAI_API_KEY",
    type="password",
    help="입력한 API 키는 현재 앱 세션에서만 사용됩니다.",
)
uploaded_file = st.file_uploader(
    "CSV 파일을 올려주세요",
    type=["csv"],
)


def read_csv(file_bytes):
    """CSV 16행을 헤더로 사용하여 데이터 행을 읽습니다."""
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = file_bytes.decode(encoding)
            csv_rows = list(csv.reader(io.StringIO(text)))

            if len(csv_rows) < 16:
                raise ValueError("CSV에 헤더로 사용할 16행이 없습니다.")

            # CSV의 물리적 16행을 컬럼명으로 사용합니다.
            raw_columns = csv_rows[15]
            columns = []
            column_counts = {}

            for index, raw_column in enumerate(raw_columns, start=1):
                column = raw_column.strip() or f"컬럼_{index}"
                column_counts[column] = column_counts.get(column, 0) + 1
                if column_counts[column] > 1:
                    column = f"{column}_{column_counts[column]}"
                columns.append(column)

            rows = []
            for row_number, values in enumerate(csv_rows[16:], start=17):
                if not any(str(value).strip() for value in values):
                    continue

                padded_values = values[: len(columns)] + [""] * (
                    len(columns) - len(values)
                )
                row = {
                    column: str(value).strip()
                    for column, value in zip(columns, padded_values)
                }
                row["__csv_row__"] = row_number
                rows.append(row)

            return columns, rows
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949로 저장해 주세요.")


def rows_to_document_records(rows):
    """CSV 각 행을 캐시 가능한 본문·메타데이터 튜플로 변환합니다."""
    records = []

    for row in rows:
        metadata = {
            column: value
            for column, value in row.items()
            if column not in (None, "__csv_row__") and value not in (None, "")
        }
        metadata["row"] = row["__csv_row__"]

        page_content = "\n".join(
            f"{column}: {value}"
            for column, value in row.items()
            if column not in (None, "__csv_row__") and value not in (None, "")
        )
        records.append((page_content, tuple(sorted(metadata.items()))))

    return tuple(records)


@st.cache_resource(show_spinner=False)
def create_rag_chain(document_records, api_key):
    """CSV 전체 행을 벡터 저장소에 등록합니다."""
    documents = [
        Document(page_content=content, metadata=dict(metadata_items))
        for content, metadata_items in document_records
    ]
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents)
    retriever = vector_store.as_retriever(
        search_kwargs={"k": min(len(documents), 20)}
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
        api_key=api_key,
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 CSV 데이터 분석 AI입니다.
        아래 Context에 포함된 행만 사용해서 질문에 답하세요.
        데이터에 없는 내용은 추측하지 마세요.
        금액 비교, 날짜 비교, 최댓값과 최솟값은 Context의 수치를 직접 비교해서 답하세요.

        Context:
        {context}

        Question:
        {input}

        답변:
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


class StreamHandler(BaseCallbackHandler):
    """생성된 토큰을 Streamlit 메인 스레드로 전달합니다."""

    def __init__(self, token_queue):
        self.token_queue = token_queue

    def on_llm_new_token(self, token, **kwargs):
        for character in token:
            self.token_queue.put(character)


def stream_chain_response(qa_chain, question, handler, container):
    """체인은 작업 스레드에서 실행하고 UI는 메인 스레드에서 갱신합니다."""
    result = {}
    error = {}

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

    streamed_text = ""
    while worker.is_alive() or not handler.token_queue.empty():
        try:
            character = handler.token_queue.get(timeout=0.05)
            streamed_text += character
            container.markdown(streamed_text)
            time.sleep(0.01)
        except Empty:
            continue

    worker.join()

    if "exception" in error:
        raise error["exception"]

    if not streamed_text:
        container.markdown(result["response"]["answer"])

    return result["response"]


if uploaded_file is not None:
    try:
        columns, rows = read_csv(uploaded_file.getvalue())
    except ValueError as error:
        st.error(str(error))
        st.stop()

    st.success(f"CSV 로딩 완료: {len(rows)}개 행")

    with st.expander("CSV 데이터 미리보기", expanded=True):
        preview_rows = [
            {column: row.get(column, "") for column in columns}
            for row in rows
        ]
        st.dataframe(preview_rows, use_container_width=True)

    if not openai_key:
        st.warning("OPENAI_API_KEY를 입력하세요.")
        st.stop()

    document_records = rows_to_document_records(rows)

    try:
        with st.spinner("CSV 데이터를 분석하고 있습니다..."):
            qa_chain = create_rag_chain(document_records, openai_key)
    except Exception as error:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {error}")
        st.stop()

    # CSV 파일이 바뀌면 기존 고정 질문의 답변을 초기화합니다.
    answer_context_id = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    if st.session_state.get("answer_context_id") != answer_context_id:
        st.session_state.answer_context_id = answer_context_id
        st.session_state.fixed_answers = {}
        st.session_state.follow_up_messages = []

    st.subheader("고정 질문")
    st.caption("질문을 누르면 해당 질문 바로 아래에 답변이 표시됩니다.")

    for question_number, question in enumerate(FIXED_QUESTIONS, start=1):
        question_key = f"fixed_question_{question_number}"
        clicked = st.button(
            f"{question_number}. {question}",
            key=question_key,
            use_container_width=True,
        )
        answer_box = st.container(border=True)

        if clicked:
            handler = StreamHandler(Queue())

            try:
                with answer_box:
                    chat_box = st.empty()
                    with st.spinner("답변 생성 중..."):
                        response = stream_chain_response(
                            qa_chain,
                            question,
                            handler,
                            chat_box,
                        )

                    references = sorted(
                        {
                            document.metadata.get("row")
                            for document in response.get("context", [])
                            if document.metadata.get("row")
                        }
                    )
                    if references:
                        st.caption(
                            "참조 CSV 행: " + ", ".join(map(str, references))
                        )

                st.session_state.fixed_answers[question_key] = {
                    "answer": response["answer"],
                    "references": references,
                }
            except Exception as error:
                with answer_box:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {error}")
        elif question_key in st.session_state.fixed_answers:
            saved_result = st.session_state.fixed_answers[question_key]
            with answer_box:
                st.write(saved_result["answer"])
                if saved_result["references"]:
                    st.caption(
                        "참조 CSV 행: "
                        + ", ".join(map(str, saved_result["references"]))
                    )
        else:
            with answer_box:
                st.caption("질문을 눌러 답변을 확인하세요.")

    # 고정 질문의 답변을 하나 이상 확인한 뒤 추가 질문 입력창을 표시합니다.
    if st.session_state.fixed_answers:
        st.divider()
        st.subheader("추가 질문")

        for message in st.session_state.follow_up_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message.get("references"):
                    st.caption(
                        "참조 CSV 행: "
                        + ", ".join(map(str, message["references"]))
                    )

        follow_up_question = st.chat_input(
            "CSV 데이터에 관해 추가로 질문하세요."
        )

        if follow_up_question:
            st.session_state.follow_up_messages.append(
                {"role": "user", "content": follow_up_question}
            )

            with st.chat_message("user"):
                st.write(follow_up_question)

            with st.chat_message("assistant"):
                chat_box = st.empty()
                handler = StreamHandler(Queue())

                try:
                    with st.spinner("답변 생성 중..."):
                        response = stream_chain_response(
                            qa_chain,
                            follow_up_question,
                            handler,
                            chat_box,
                        )

                    references = sorted(
                        {
                            document.metadata.get("row")
                            for document in response.get("context", [])
                            if document.metadata.get("row")
                        }
                    )
                    if references:
                        st.caption(
                            "참조 CSV 행: " + ", ".join(map(str, references))
                        )

                    st.session_state.follow_up_messages.append(
                        {
                            "role": "assistant",
                            "content": response["answer"],
                            "references": references,
                        }
                    )
                except Exception as error:
                    st.error(f"답변 생성 중 오류가 발생했습니다: {error}")
