import os
import pathlib
from typing import List, Annotated

# ----------------------------------------------------------------
# Gemini API 사용을 위한 라이브러리
# pip install google-generativeai
import google.generativeai as genai

# ----------------------------------------------------------------

from rag.base import RetrievalChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ⭐️ 시작 전 GOOGLE_API_KEY 환경 변수 설정이 필요합니다.
# 예: os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
except AttributeError as e:
    print(
        "⚠️ 경고: GOOGLE_API_KEY가 설정되지 않았습니다. Gemini API를 사용할 수 없습니다."
    )


class GeminiPDFRetrievalChain(RetrievalChain):
    """
    base.py의 RetrievalChain을 상속받아 Gemini를 사용하여 PDF를 처리하는 클래스입니다.
    """

    def __init__(self, source_uri: Annotated[str, "Source URI"]):
        """
        초기화 메서드입니다. 처리할 PDF 파일의 경로(URI)를 받습니다.
        """
        self.source_uri = source_uri
        self.k = 10  # 검색할 문서의 수를 10으로 설정

    def load_documents(self, source_uris: List[str]) -> List[Document]:
        """
        @override
        Gemini API를 사용하여 PDF 파일에서 텍스트를 추출하고 LangChain Document 객체로 변환합니다.
        """
        docs = []
        for file_path in source_uris:
            print(f"Gemini로 '{file_path}' 파일 처리 시작...")
            uploaded_file = None  # finally 블록에서 사용하기 위해 초기화
            try:
                # 1. Gemini API에 PDF 파일 업로드
                uploaded_file = genai.upload_file(
                    path=file_path, display_name=os.path.basename(file_path)
                )

                # 2. Gemini에 요청할 프롬프트 정의
                prompt = """당신은 PDF 문서 분석 전문가입니다.
                            주어진 PDF 파일의 각 페이지를 순서대로 분석하여, 페이지별로 내용을 구분해서 추출해주세요.

                            출력 형식:
                            [PAGE 1]
                            (1페이지의 텍스트 내용)

                            [PAGE 2]
                            (2페이지의 텍스트 내용)

                            ... (이런 식으로 계속)

                            각 페이지에서:
                            - 텍스트와 표(table) 형식의 데이터를 모두 추출
                            - 차트나 이미지에 대한 설명은 제외
                            - 순수 텍스트와 표의 내용만 정확하게 추출

                            반드시 [PAGE N] 마커로 각 페이지를 구분해주세요.
                         """

                # 3. Gemini-2.5-Flash 모델로 콘텐츠 생성 요청
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content([prompt, uploaded_file])

                # 4. 추출된 텍스트를 페이지별로 분할하여 Document 객체 생성
                import re
                page_pattern = r'\[PAGE\s+(\d+)\](.*?)(?=\[PAGE\s+\d+\]|$)'
                matches = re.findall(page_pattern, response.text, re.DOTALL)

                if matches:
                    # 페이지별로 구분된 경우
                    for page_num, content in matches:
                        doc = Document(
                            metadata={
                                "source": file_path,
                                "page": int(page_num)
                            },
                            page_content=content.strip(),
                        )
                        docs.append(doc)
                else:
                    # 페이지 구분이 없는 경우 (fallback)
                    doc = Document(
                        metadata={
                            "source": file_path,
                            "page": 1  # 기본값으로 1페이지 설정
                        },
                        page_content=response.text,
                    )
                    docs.append(doc)
                print(f"✅ '{file_path}' 파일 처리 완료.")

            except Exception as e:
                print(f"❌ '{file_path}' 파일 처리 중 오류 발생: {e}")
            finally:
                # 5. 작업 완료 후 업로드된 파일 삭제 (리소스 정리)
                if uploaded_file:
                    genai.delete_file(uploaded_file.name)
                    print(f"'{uploaded_file.display_name}' 원격 파일 삭제 완료.")

        # return response  # 이 줄은 오류였습니다
        return docs

    def create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        @override
        추출된 텍스트를 분할할 Text Splitter를 생성합니다.
        pdf.py와 동일한 분할 전략을 사용합니다.
        """
        return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
