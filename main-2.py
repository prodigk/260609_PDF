# pip install -U langchain-text-splitters

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF 파일 경로 설정 및 로더 초기화
loader = PyPDFLoader("NAVER_20260607.pdf")

# PDF 페이지 로드 및 분할
pages = loader.load_and_split()

# split 단계 (텍스트 청크 쪼개기)
# LLM이 처리하기 좋게 문서를 더 작은 단위(chunk)로 잘게 쪼갠다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300, # 하나의 텍스트 조각(Chunk)에 들어갈 최대 글자 수
  
    chunk_overlap = 20, # 앞뒤 텍스트 조각 간에 겹칠 글자 수, 문맥이 끊기는 것을 방지하기 위해 보통 10~20% 정도 겹치게 설정함

    length_function = len, # 텍스트의 길이를 계산하는 함수, 기본적으로는 글자 수를 세는 len 함수를 사용하지만, 단어 수나 토큰 수로 계산하는 다른 함수를 사용할 수도 있음

    is_separator_regex = False, # 구분자(separator)가 정규 표현식으로 표현할지 판단, False로 설정하면 단순 문자열로 구분자를 사용하고, True로 설정하면 정규 표현식으로 구분자를 사용함
)

# 설정한 chunk_size(300자) 기준에 맞춰 최종 텍스트 조각들로 분할한다.
texts = text_splitter.split_documents(pages)

if texts:
    print("--- [첫 번째 텍스트 조각(Chunk) 객체 출력] ---")
    print(texts[0])
    
    print("\n--- [첫 번째 조각의 실제 텍스트 내용만 출력] ---")
    print(texts[0].page_content)
else:
    print("분할된 텍스트 조각이 없습니다. PDF 파일 내용을 확인해 주세요.")