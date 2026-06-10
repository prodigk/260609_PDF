# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv
# pip install --upgrade langchain langchain-openai langchain-chroma pypdf python-dotenv

from dotenv import load_dotenv
# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader("NAVER_20260607.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap = 20,         # 청크 간에 겹칠 글자 수 (문맥 단절 방지)
    length_function = len,      # 길이를 측정할 함수 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호(separator)를 정규표현식으로 해석할지 여부
)

texts = text_splitter.split_documents(pages)
# 임베딩 모델 생성
# OpenAIEmbeddings는 OpenAI의 임베딩 모델을 사용하여 텍스트를 벡터로 변환하는 클래스입니다.
# 필요에 따라 model="text-embedding-3-small" 같은 특정 model을 지정할 수 있다.
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
#embeddings_model = OpenAIEmbeddings()

# 벡터 데이터베이스 생성
# Chroma.from_documents() 메서드는 텍스트 문서와 임베딩 모델을 사용하여 벡터 데이터베이스를 생성하는 함수입니다.
# 현재 코드는 메모리 상에 임시 저장하는 방식입니다.
db = Chroma.from_documents(texts, embeddings_model)

