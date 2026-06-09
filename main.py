# uv pip install -U langchain-community pypdf

from langchain_community.document_loaders import PyPDFLoader

# PDF 파일 경로 설정 and 로더 초기화
file_path = "NAVER_20260607.pdf"
loader = PyPDFLoader(file_path)

pages = loader.load_and_split()

# 데이터 확인 및 출력 
if len(pages) > 1:
    print("-------------[두 번째 페이지 객체 전체 출력]-------------")
    print(pages[1])
    
    print("-------------[두 번째 페이지 실제 텍스트 내용만 출력]-------------")
    print(pages[1].page_content)
else:
    print(f"PDF 문서에 페이지가 {len(pages)}개 있습니다.")


#print(pages)
#print(pages[0])