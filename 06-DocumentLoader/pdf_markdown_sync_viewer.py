import os
import gradio as gr
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
import fitz  # PyMuPDF for PDF display
import base64
import json
from pathlib import Path
import pickle
import re

# 환경 변수 로드
load_dotenv()

# jupyter 환경에서 asyncio를 사용할 수 있도록 설정
nest_asyncio.apply()


class DocumentMarkdownViewer:
    def __init__(self):
        self.current_file_path = None
        self.file_type = None
        self.file_pages = []  # 원본 파일의 페이지들 (이미지 또는 텍스트)
        self.markdown_pages = []
        self.current_page = 0
        self.total_pages = 0
        
        # 지원하는 파일 형식별 분류
        self.image_formats = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.tiff', '.webp'}
        self.web_formats = {'.html', '.htm'}  # 웹 문서는 iframe으로 표시 가능
        self.text_formats = {'.txt', '.xml', '.md', '.json', '.csv'}  # 텍스트는 직접 표시 가능
        self.document_formats = {'.doc', '.docx', '.docm', '.dot', '.dotm', '.rtf', '.hwp'}
        self.presentation_formats = {'.ppt', '.pptx', '.pptm', '.pot', '.potm', '.potx', '.key'}
        self.spreadsheet_formats = {'.xlsx', '.xls', '.xlsm', '.xlsb', '.numbers', '.ods'}
        self.ebook_formats = {'.epub'}
        
    def get_file_type_category(self, file_path):
        """파일 확장자로 파일 형식 카테고리 결정"""
        ext = Path(file_path).suffix.lower()
        
        if ext in self.image_formats:
            return 'image'
        elif ext in self.web_formats:
            return 'web'
        elif ext in self.text_formats:
            return 'text'
        elif ext in self.document_formats:
            return 'document'
        elif ext in self.presentation_formats:
            return 'presentation'
        elif ext in self.spreadsheet_formats:
            return 'spreadsheet'
        elif ext in self.ebook_formats:
            return 'ebook'
        else:
            return 'unknown'
        
    def file_to_pages(self, file_path):
        """파일을 페이지별로 처리 (형식에 따라 다른 방식 적용)"""
        file_type = self.get_file_type_category(file_path)
        self.file_type = file_type
        
        if file_type == 'image' and Path(file_path).suffix.lower() == '.pdf':
            return self.pdf_to_page_images(file_path)
        elif file_type == 'image':
            return self.image_to_base64(file_path)
        elif file_type == 'web':
            return self.web_to_display(file_path)
        elif file_type == 'text':
            return self.text_to_display(file_path)
        else:
            # 기타 파일들은 파일 정보 표시
            return self.create_file_info_display(file_path)
    
    def pdf_to_page_images(self, pdf_path):
        """PDF 파일을 페이지별 이미지로 변환"""
        try:
            pdf_document = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                # 페이지를 이미지로 변환
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # base64로 인코딩
                img_base64 = base64.b64encode(img_data).decode()
                pages.append(img_base64)
            
            pdf_document.close()
            return pages
        
        except Exception as e:
            print(f"PDF 이미지 변환 오류: {str(e)}")
            return []
    
    def image_to_base64(self, image_path):
        """이미지 파일을 base64로 변환"""
        try:
            with open(image_path, 'rb') as image_file:
                img_data = image_file.read()
                img_base64 = base64.b64encode(img_data).decode()
                return [img_base64]  # 단일 이미지를 리스트로 반환
        except Exception as e:
            print(f"이미지 변환 오류: {str(e)}")
            return []
    
    def web_to_display(self, web_path):
        """웹 문서(HTML)를 표시하기 위한 HTML 생성"""
        try:
            with open(web_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # 보안을 위해 iframe 대신 내용을 직접 표시
            display_html = f"""
            <div style='border: 1px solid #ddd; padding: 20px; background-color: white; border-radius: 8px; max-height: 600px; overflow-y: auto;'>
                <div style='background-color: #f0f0f0; padding: 10px; margin-bottom: 15px; border-radius: 4px; text-align: center;'>
                    <strong>📄 HTML 문서: {Path(web_path).name}</strong>
                </div>
                <div style='border: 1px solid #ccc; padding: 15px; background-color: #fafafa; border-radius: 4px;'>
                    {html_content}
                </div>
            </div>
            """
            return [display_html]
        except Exception as e:
            print(f"웹 문서 로드 오류: {str(e)}")
            return [f"<p style='color: red;'>웹 문서를 로드할 수 없습니다: {str(e)}</p>"]
    
    def text_to_display(self, text_path):
        """텍스트 파일을 표시하기 위한 HTML 생성"""
        try:
            # 다양한 인코딩으로 시도
            encodings = ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr', 'latin-1']
            content = None
            used_encoding = None
            
            for encoding in encodings:
                try:
                    with open(text_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    used_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return [f"<p style='color: red;'>텍스트 파일을 읽을 수 없습니다 (인코딩 문제)</p>"]
            
            # 파일 확장자에 따른 처리
            ext = Path(text_path).suffix.lower()
            if ext == '.json':
                try:
                    import json
                    # JSON 형식으로 pretty print
                    json_obj = json.loads(content)
                    content = json.dumps(json_obj, indent=2, ensure_ascii=False)
                    lang_class = "json"
                except:
                    lang_class = "text"
            elif ext == '.xml':
                lang_class = "xml"
            elif ext == '.csv':
                lang_class = "csv"
            elif ext == '.md':
                lang_class = "markdown"
            else:
                lang_class = "text"
            
            # 텍스트 길이 제한 (너무 긴 경우)
            if len(content) > 50000:  # 50KB 제한
                content = content[:50000] + "\n\n... (파일이 너무 커서 일부만 표시됩니다)"
            
            display_html = f"""
            <div style='border: 1px solid #ddd; padding: 20px; background-color: white; border-radius: 8px;'>
                <div style='background-color: #f0f0f0; padding: 10px; margin-bottom: 15px; border-radius: 4px; text-align: center;'>
                    <strong>📝 {Path(text_path).name}</strong>
                    <br><small style='color: #666;'>인코딩: {used_encoding}</small>
                </div>
                <div style='border: 1px solid #ccc; padding: 15px; background-color: #f8f8f8; border-radius: 4px; max-height: 500px; overflow-y: auto;'>
                    <pre style='margin: 0; white-space: pre-wrap; word-wrap: break-word; font-family: "Courier New", monospace; font-size: 12px; line-height: 1.4;'><code class='{lang_class}'>{content}</code></pre>
                </div>
            </div>
            """
            return [display_html]
        except Exception as e:
            print(f"텍스트 파일 로드 오류: {str(e)}")
            return [f"<p style='color: red;'>텍스트 파일을 로드할 수 없습니다: {str(e)}</p>"]
    
    def create_file_info_display(self, file_path):
        """PDF가 아닌 파일들을 위한 정보 표시 생성"""
        file_path = Path(file_path)
        file_type = self.get_file_type_category(str(file_path))
        
        # 파일 형식별 아이콘
        icons = {
            'document': '📝',
            'presentation': '🎯', 
            'spreadsheet': '📊',
            'text': '📄',
            'web': '🌐',
            'ebook': '📚',
            'unknown': '📁'
        }
        
        # 파일 형식별 설명
        type_descriptions = {
            'document': 'Word 문서 등의 텍스트 문서',
            'presentation': 'PowerPoint 프레젠테이션',
            'spreadsheet': 'Excel 스프레드시트',
            'text': '텍스트 파일 (원본 내용 표시 가능)',
            'web': 'HTML 웹 문서 (원본 내용 표시 가능)',
            'ebook': '전자책',
            'unknown': '기타 문서'
        }
        
        icon = icons.get(file_type, '📁')
        description = type_descriptions.get(file_type, '문서 파일')
        
        try:
            file_size = file_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            size_str = f"{size_mb:.2f} MB" if size_mb > 1 else f"{file_size / 1024:.2f} KB"
        except:
            size_str = "크기 불명"
        
        # PPT 파일의 경우 특별한 안내
        ppt_note = ""
        if file_type == 'presentation':
            ppt_note = """
            <div style='background-color: #e8f4fd; padding: 15px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #007acc;'>
                <p style='color: #0056b3; font-weight: bold; margin: 0 0 10px 0;'>📋 PowerPoint 파일 안내</p>
                <p style='color: #0056b3; font-size: 14px; margin: 0; line-height: 1.4;'>
                    • 슬라이드 내용은 우측 Markdown 패널에서 확인할 수 있습니다<br>
                    • LlamaParse가 텍스트, 표, 이미지 설명을 자동 추출합니다<br>
                    • 원본 시각적 표시는 현재 지원하지 않지만 내용은 완전히 변환됩니다
                </p>
            </div>
            """
        
        info_html = f"""
        <div style='text-align: center; padding: 40px; border: 2px dashed #ccc; background-color: #f9f9f9; border-radius: 10px;'>
            <div style='font-size: 64px; margin-bottom: 20px;'>{icon}</div>
            <h3 style='color: #333; margin-bottom: 10px;'>{file_path.name}</h3>
            <p style='color: #666; margin-bottom: 5px; font-weight: bold;'>{description}</p>
            <p style='color: #888; margin-bottom: 5px;'>파일 형식: {file_type.upper()} ({file_path.suffix.upper()})</p>
            <p style='color: #888; margin-bottom: 20px;'>파일 크기: {size_str}</p>
            <p style='color: #007acc; font-size: 16px; font-weight: bold;'>✨ LlamaParse로 지능적 변환 중...</p>
            <p style='color: #999; font-size: 14px;'>변환된 내용은 우측 Markdown 패널에서 확인하세요</p>
            {ppt_note}
        </div>
        """
        
        return [info_html]
    
    def parse_file_to_markdown_pages(self, file_path):
        """다양한 파일 형식을 페이지별 Markdown으로 변환"""
        try:
            file_type = self.get_file_type_category(file_path)
            
            # 파일 형식별 system prompt 설정
            if file_type == 'presentation':
                system_prompt = (
                    "You are parsing a presentation document. Please extract all content including text, tables, and slide structure in markdown format. "
                    "Preserve the structure and formatting as much as possible. "
                    "Mark slide breaks clearly with '---PAGE_BREAK---' marker. "
                    "Include any chart or image descriptions if present."
                )
            elif file_type == 'spreadsheet':
                system_prompt = (
                    "You are parsing a spreadsheet document. Please extract all content including tables, data, and sheet structure in markdown format. "
                    "Preserve table structures using markdown table format. "
                    "Mark sheet breaks clearly with '---PAGE_BREAK---' marker if multiple sheets exist."
                )
            elif file_type == 'document':
                system_prompt = (
                    "You are parsing a document file. Please extract all content including text, tables, and document structure in markdown format. "
                    "Preserve the structure and formatting as much as possible. "
                    "Mark page breaks clearly with '---PAGE_BREAK---' marker."
                )
            else:
                # 기본 prompt (PDF, 이미지, 기타 등)
                system_prompt = (
                    "You are parsing a document. Please extract all content including tables in markdown format. "
                    "Preserve the structure and formatting as much as possible. "
                    "Mark page breaks clearly with '---PAGE_BREAK---' marker."
                )
            
            # LlamaParse 설정
            parser = LlamaParse(
                use_vendor_multimodal_model=True,
                vendor_multimodal_model_name="openai-gpt4o",
                vendor_multimodal_api_key=os.environ.get("OPENAI_API_KEY"),
                result_type="markdown",
                language="ko",
                system_prompt=system_prompt,
            )
            
            # 파일 파싱
            parsed_docs = parser.load_data(file_path=file_path)
            
            # 페이지별로 분리
            markdown_pages = []
            for doc in parsed_docs:
                # LangChain 형식으로 변환
                content = doc.to_langchain_format().page_content
                
                # 페이지 구분자로 분리 (LlamaParse가 페이지를 구분하는 경우)
                pages = content.split('---PAGE_BREAK---')
                if len(pages) == 1:
                    # 페이지 구분자가 없으면 전체를 하나의 페이지로
                    markdown_pages.append(content)
                else:
                    markdown_pages.extend(pages)
            
            return markdown_pages
        
        except Exception as e:
            print(f"Markdown 변환 오류: {str(e)}")
            return []
    
    def save_conversion(self, file_path, file_pages, markdown_pages):
        """변환 결과를 MD 파일로 저장"""
        try:
            # 저장 디렉토리 생성
            save_dir = Path("conversions")
            save_dir.mkdir(exist_ok=True)
            
            # 파일명 기반으로 저장 (중복 방지)
            base_name = Path(file_path).stem
            md_path = save_dir / f"{base_name}.md"
            
            # 중복 파일명 처리
            counter = 1
            while md_path.exists():
                md_path = save_dir / f"{base_name}_{counter}.md"
                counter += 1
            
            # 파일 형식 정보
            file_type = self.get_file_type_category(file_path)
            file_ext = Path(file_path).suffix.upper()
            
            # Markdown 파일로 저장 (원본 파일 경로 정보 포함)
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(f"# {base_name}\n\n")
                f.write(f"원본 파일 경로: {file_path}\n")
                f.write(f"파일 형식: {file_type} ({file_ext})\n")
                f.write(f"변환 날짜: {Path(file_path).stat().st_mtime}\n")
                f.write(f"총 페이지: {len(markdown_pages)}\n\n")
                f.write("<!-- FILE_PATH_MARKER -->\n\n")
                
                for i, page in enumerate(markdown_pages):
                    f.write(f"## 페이지 {i+1}\n\n")
                    f.write(page)
                    f.write(f"\n\n{'='*50}\n\n")
            
            return str(md_path)
        
        except Exception as e:
            print(f"저장 오류: {str(e)}")
            return None
    
    def get_available_conversions(self):
        """저장된 MD 파일 목록 가져오기"""
        save_dir = Path("conversions")
        if not save_dir.exists():
            return []
        
        files = list(save_dir.glob("*.md"))
        return [str(f) for f in files]
    
    def get_all_file_html(self):
        """모든 파일 페이지를 연속된 HTML로 반환"""
        if not self.file_pages:
            return "<p style='text-align: center; color: #666;'>파일이 로드되지 않았습니다.</p>"
        
        html_parts = []
        try:
            for i, page_content in enumerate(self.file_pages):
                if not page_content:  # 빈 내용 체크
                    html_parts.append(f'''
                        <div class="file-page" data-page="{i}">
                            <div class="page-number">페이지 {i+1}</div>
                            <p style='text-align: center; color: #666;'>빈 페이지</p>
                        </div>
                    ''')
                elif page_content.startswith('<div'):
                    # 이미 HTML 형식인 경우 (파일 정보 디스플레이)
                    html_parts.append(f'''
                        <div class="file-page" data-page="{i}">
                            <div class="page-number">페이지 {i+1}</div>
                            {page_content}
                        </div>
                    ''')
                else:
                    # 이미지 base64인 경우
                    # 파일 확장자에 따라 적절한 MIME 타입 결정
                    if self.file_type == 'image' and hasattr(self, 'current_file_path') and self.current_file_path:
                        ext = Path(self.current_file_path).suffix.lower()
                        if ext == '.pdf':
                            mime_type = "image/png"  # PDF는 PNG로 변환됨
                        elif ext in ['.jpg', '.jpeg']:
                            mime_type = "image/jpeg"
                        elif ext == '.png':
                            mime_type = "image/png"
                        elif ext == '.gif':
                            mime_type = "image/gif"
                        elif ext == '.bmp':
                            mime_type = "image/bmp"
                        elif ext == '.svg':
                            mime_type = "image/svg+xml"
                        elif ext in ['.tiff', '.tif']:
                            mime_type = "image/tiff"
                        elif ext == '.webp':
                            mime_type = "image/webp"
                        else:
                            mime_type = "image/png"  # 기본값
                    else:
                        mime_type = "image/png"  # 기본값
                    
                    html_parts.append(f'''
                        <div class="file-page" data-page="{i}">
                            <div class="page-number">페이지 {i+1}</div>
                            <img src="data:{mime_type};base64,{page_content}" style="width: 100%; margin-bottom: 20px;" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                            <p style='text-align: center; color: #666; display: none;'>이미지를 표시할 수 없습니다</p>
                        </div>
                    ''')
                    
        except Exception as e:
            print(f"[DEBUG] HTML 생성 중 오류: {e}")
            html_parts.append(f"<p style='color: red;'>HTML 생성 오류: {str(e)}</p>")
            
        return ''.join(html_parts) if html_parts else "<p style='text-align: center; color: #666;'>표시할 내용이 없습니다.</p>"
    
    def get_all_markdown_content(self):
        """모든 Markdown 페이지를 연속된 내용으로 반환"""
        if not self.markdown_pages:
            return "Markdown 내용이 없습니다."
        
        content_parts = []
        try:
            for i, page_content in enumerate(self.markdown_pages):
                if page_content:  # 빈 내용이 아닌 경우만
                    content_parts.append(f"## 페이지 {i+1}\n\n{page_content}\n\n---\n\n")
                else:
                    content_parts.append(f"## 페이지 {i+1}\n\n*(빈 페이지)*\n\n---\n\n")
        except Exception as e:
            print(f"[DEBUG] Markdown 내용 생성 중 오류: {e}")
            content_parts.append(f"오류: {str(e)}")
        
        return ''.join(content_parts) if content_parts else "표시할 Markdown 내용이 없습니다."


# 전역 뷰어 인스턴스
viewer = DocumentMarkdownViewer()


def process_file(input_file):
    """다양한 파일 형식 처리 및 변환"""
    if input_file is None:
        return (
            "파일을 업로드해주세요.",
            "",
            "파일을 업로드해주세요."
        )
    
    try:
        # 파일 형식 확인
        file_type = viewer.get_file_type_category(input_file.name)
        file_ext = Path(input_file.name).suffix.upper()
        print(f"[DEBUG] 파일 형식: {file_type}, 확장자: {file_ext}")
        
        # 파일을 페이지별로 처리 (형식에 따라 다른 방식)
        try:
            viewer.file_pages = viewer.file_to_pages(input_file.name)
            print(f"[DEBUG] 파일 페이지 수: {len(viewer.file_pages)}")
        except Exception as e:
            print(f"[DEBUG] 파일 페이지 처리 오류: {e}")
            viewer.file_pages = []
        
        # Markdown으로 변환
        try:
            viewer.markdown_pages = viewer.parse_file_to_markdown_pages(input_file.name)
            print(f"[DEBUG] Markdown 페이지 수: {len(viewer.markdown_pages)}")
        except Exception as e:
            print(f"[DEBUG] Markdown 변환 오류: {e}")
            viewer.markdown_pages = []
        
        # 최소 1개의 페이지는 있어야 함
        if not viewer.file_pages:
            viewer.file_pages = [f"<p style='color: red;'>파일을 처리할 수 없습니다: {file_type.upper()} {file_ext}</p>"]
        if not viewer.markdown_pages:
            viewer.markdown_pages = ["파일을 변환할 수 없습니다."]
        
        # 페이지 수가 다른 경우 조정
        if len(viewer.markdown_pages) < len(viewer.file_pages):
            # Markdown 페이지가 적으면 빈 페이지 추가
            while len(viewer.markdown_pages) < len(viewer.file_pages):
                viewer.markdown_pages.append("*(이 페이지의 Markdown 변환 내용이 없습니다)*")
        elif len(viewer.markdown_pages) > len(viewer.file_pages):
            # Markdown이 더 많으면 합치기
            if len(viewer.file_pages) > 0:  # 안전성 확인
                combined = "\n\n".join(viewer.markdown_pages[len(viewer.file_pages):])
                viewer.markdown_pages = viewer.markdown_pages[:len(viewer.file_pages)]
                if viewer.markdown_pages:  # 빈 리스트가 아닌지 확인
                    viewer.markdown_pages[-1] += "\n\n" + combined
        
        viewer.total_pages = max(len(viewer.file_pages), 1)  # 최소 1페이지
        viewer.current_page = 0
        viewer.current_file_path = input_file.name
        
        # 변환 결과 저장
        try:
            save_path = viewer.save_conversion(
                input_file.name, 
                viewer.file_pages, 
                viewer.markdown_pages
            )
        except Exception as e:
            print(f"[DEBUG] 저장 오류: {e}")
            save_path = "저장 실패"
        
        # 모든 페이지를 스크롤 방식으로 표시
        if viewer.total_pages > 0:
            try:
                file_html = viewer.get_all_file_html()
                markdown_content = viewer.get_all_markdown_content()
                
                return (
                    file_html,
                    markdown_content,
                    f"✅ 변환 완료! {file_type.upper()} {file_ext} 파일, 총 {viewer.total_pages}페이지 저장됨 (위치: {save_path})"
                )
            except Exception as e:
                print(f"[DEBUG] 표시 생성 오류: {e}")
                return (
                    f"<p style='color: red;'>표시 오류: {str(e)}</p>",
                    "표시 오류가 발생했습니다.",
                    f"⚠️ 부분 변환 완료: {file_type.upper()} {file_ext}"
                )
        else:
            return (
                "파일을 불러올 수 없습니다.",
                "",
                "❌ 변환 실패"
            )
    
    except Exception as e:
        return (
            f"<p style='color: red;'>오류: {str(e)}</p>",
            f"오류: {str(e)}",
            f"❌ 오류 발생: {str(e)}"
        )


def refresh_display():
    """전체 내용을 새로고침"""
    if not viewer.file_pages or not viewer.markdown_pages:
        return "파일을 먼저 업로드하세요.", "내용이 없습니다."
    
    file_html = viewer.get_all_file_html()
    markdown_content = viewer.get_all_markdown_content()
    
    return file_html, markdown_content


def load_saved_conversion(file_path):
    """저장된 MD 파일과 원본 PDF 불러오기"""
    if not file_path:
        return (
            "파일을 선택하세요.",
            "",
            "파일을 선택하세요."
        )
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # MD 파일에서 원본 파일 경로 추출 (새 형식과 레거시 형식 모두 지원)
        original_file_path = None
        
        # 새 형식: "원본 파일 경로: " 패턴
        new_pattern = r'원본 파일 경로: (.+)'
        new_match = re.search(new_pattern, content)
        
        # 레거시 형식: "원본 PDF 경로: " 패턴
        legacy_pattern = r'원본 PDF 경로: (.+)'  
        legacy_match = re.search(legacy_pattern, content)
        
        if new_match:
            original_file_path = new_match.group(1).strip()
            print(f"[DEBUG] 원본 파일 경로 찾음: {original_file_path}")
        elif legacy_match:
            original_file_path = legacy_match.group(1).strip()
            print(f"[DEBUG] 레거시 PDF 경로 찾음: {original_file_path}")
        else:
            print(f"[DEBUG] 파일 경로를 찾을 수 없음 (레거시 파일). 자동 검색을 시도합니다.")
            # 레거시 파일의 경우 같은 이름의 파일을 자동으로 찾기
            md_filename = Path(file_path).stem
            
            # 검색할 경로들
            search_paths = [
                Path(file_path).parent,  # 같은 폴더 (conversions)
                Path(file_path).parent.parent / "data",  # data 폴더
                Path(file_path).parent.parent,  # 상위 폴더
            ]
            
            # 다양한 확장자로 시도
            extensions = ['.pdf', '.docx', '.pptx', '.xlsx', '.doc', '.ppt', '.xls', '.hwp']
            
            for search_dir in search_paths:
                if search_dir.exists():
                    for ext in extensions:
                        potential_file = search_dir / f"{md_filename}{ext}"
                        if potential_file.exists():
                            original_file_path = str(potential_file)
                            print(f"[DEBUG] 자동으로 찾은 파일: {original_file_path}")
                            break
                    if original_file_path:
                        break
        
        # 원본 파일이 존재하는지 확인하고 로드
        viewer.file_pages = []
        viewer.file_type = None
        if original_file_path:
            if Path(original_file_path).exists():
                print(f"[DEBUG] 원본 파일 존재 확인됨: {original_file_path}")
                try:
                    # 파일을 적절한 형식으로 변환
                    viewer.file_pages = viewer.file_to_pages(original_file_path)
                    viewer.current_file_path = original_file_path
                    print(f"[DEBUG] 파일 페이지 로드 완료: {len(viewer.file_pages)}페이지")
                except Exception as file_error:
                    print(f"[DEBUG] 파일 로드 오류: {file_error}")
                    viewer.file_pages = []
            else:
                print(f"[DEBUG] 원본 파일이 존재하지 않음: {original_file_path}")
                original_file_path = None  # 파일이 없으면 None으로 설정
        else:
            print(f"[DEBUG] 원본 파일을 찾을 수 없음")
        
        # MD 파일에서 페이지별 내용 추출
        viewer.markdown_pages = []
        
        # "## 페이지 X" 패턴으로 분리
        page_pattern = r'## 페이지 (\d+)'
        pages = re.split(page_pattern, content)
        
        if len(pages) > 1:
            # 첫 번째 부분(제목 등)은 제외하고 페이지별로 처리
            for i in range(2, len(pages), 2):  # 페이지 번호와 내용이 번갈아 나타남
                if i < len(pages):
                    page_content = pages[i].strip()
                    # 구분선 제거
                    page_content = page_content.replace('='*50, '').strip()
                    viewer.markdown_pages.append(page_content)
        else:
            # 페이지 구분이 없으면 전체 내용을 하나로
            viewer.markdown_pages = [content]
        
        # 페이지 수 맞추기
        if viewer.file_pages and viewer.markdown_pages:
            max_pages = max(len(viewer.file_pages), len(viewer.markdown_pages))
            
            # 파일 페이지가 부족하면 빈 페이지 추가
            while len(viewer.file_pages) < max_pages:
                viewer.file_pages.append("")
                
            # Markdown 페이지가 부족하면 빈 페이지 추가
            while len(viewer.markdown_pages) < max_pages:
                viewer.markdown_pages.append("*(이 페이지의 내용이 없습니다)*")
                
            viewer.total_pages = max_pages
        else:
            viewer.total_pages = max(len(viewer.file_pages), len(viewer.markdown_pages))
        
        viewer.current_page = 0
        
        # 결과 생성
        if viewer.file_pages and any(viewer.file_pages):  # 원본 파일이 있으면
            file_html = viewer.get_all_file_html()
            file_type = viewer.get_file_type_category(original_file_path) if original_file_path else "파일"
            file_ext = Path(original_file_path).suffix.upper() if original_file_path else ""
            
            if new_match or legacy_match:  # 원래 경로에서 찾은 경우
                status_msg = f"✅ 불러오기 완료: {Path(file_path).name} ({file_type.upper()} {file_ext} + Markdown, 총 {viewer.total_pages}페이지)"
            else:  # 자동 검색으로 찾은 경우
                status_msg = f"✅ 불러오기 완료: {Path(file_path).name} ({file_type.upper()} {file_ext} 자동 검색 성공 + Markdown, 총 {viewer.total_pages}페이지)"
        else:  # 원본 파일이 없거나 로드 실패시
            if original_file_path:
                file_html = f"<p style='text-align: center; color: #666; font-size: 18px;'>⚠️ 원본 파일을 찾을 수 없습니다<br><small>{original_file_path}</small></p>"
                status_msg = f"⚠️ MD 파일만 불러옴: {Path(file_path).name} (원본 파일 없음, 총 {viewer.total_pages}페이지)"
            else:
                # 파일 경로 정보가 아예 없는 경우 (레거시 파일이고 파일도 못 찾은 경우)
                file_html = "<p style='text-align: center; color: #666; font-size: 18px;'>📄 레거시 MD 파일입니다<br><small>원본 파일을 찾을 수 없어 Markdown만 표시됩니다</small></p>"
                status_msg = f"ℹ️ 레거시 MD 파일 불러옴: {Path(file_path).name} (원본 파일 없음, 총 {viewer.total_pages}페이지)"
        
        markdown_content = viewer.get_all_markdown_content()
        
        return (
            file_html,
            markdown_content,
            status_msg
        )
        
    except Exception as e:
        return (
            f"<p style='color: red;'>오류: {str(e)}</p>",
            f"오류: {str(e)}",
            f"❌ 파일 불러오기 실패: {str(e)}"
        )


def load_markdown_file(md_file):
    """Markdown 파일만 불러와서 메인 뷰어에 전체 내용으로 표시"""
    if not md_file:
        return (
            "Markdown 파일을 선택하세요.",
            "",
            "Markdown 파일을 선택하세요."
        )
    
    try:
        with open(md_file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 페이지별로 분리 시도
        # 방법 1: "--- 페이지 X ---" 패턴
        page_pattern = r'---\s*페이지\s*(\d+)\s*---'
        pages = re.split(page_pattern, content)
        
        if len(pages) > 1:
            viewer.markdown_pages = []
            for i in range(1, len(pages), 2):
                if i+1 < len(pages):
                    viewer.markdown_pages.append(pages[i+1].strip())
        else:
            # 방법 2: 섹션 구분자로 분리
            sections = content.split('\n---\n')
            if len(sections) > 1:
                viewer.markdown_pages = sections
            else:
                # 방법 3: 큰 단락으로 분리
                paragraphs = content.split('\n\n\n')
                viewer.markdown_pages = paragraphs if len(paragraphs) > 1 else [content]
        
        # 원본 파일이 로드되어 있으면 페이지 수 맞추기
        if viewer.file_pages:
            if len(viewer.markdown_pages) < len(viewer.file_pages):
                while len(viewer.markdown_pages) < len(viewer.file_pages):
                    viewer.markdown_pages.append("*(이 페이지의 Markdown이 없습니다)*")
            elif len(viewer.markdown_pages) > len(viewer.file_pages):
                # 초과분은 마지막 페이지에 합치기
                excess = viewer.markdown_pages[len(viewer.file_pages):]
                viewer.markdown_pages = viewer.markdown_pages[:len(viewer.file_pages)]
                viewer.markdown_pages[-1] += "\n\n" + "\n\n".join(excess)
            
            viewer.total_pages = len(viewer.file_pages)
        else:
            # 원본 파일이 없으면 Markdown 페이지 수로 설정
            viewer.total_pages = len(viewer.markdown_pages)
            viewer.file_pages = [""] * viewer.total_pages  # 빈 파일 페이지
        
        viewer.current_page = 0
        
        # 모든 페이지를 스크롤 방식으로 표시
        if viewer.file_pages and viewer.file_pages[0]:
            file_html = viewer.get_all_file_html()
        else:
            file_html = "<p style='text-align: center; color: #666;'>원본 파일이 로드되지 않았습니다.</p>"
        
        markdown_content = viewer.get_all_markdown_content()
        
        return (
            file_html,
            markdown_content,
            f"✅ Markdown 로드 완료: {Path(md_file.name).name} (총 {viewer.total_pages}페이지)"
        )
    
    except Exception as e:
        return (
            f"<p style='color: red;'>오류: {str(e)}</p>",
            f"오류: {str(e)}",
            f"❌ 파일 로드 실패: {str(e)}"
        )


def get_saved_files():
    """저장된 파일 목록 가져오기"""
    files = viewer.get_available_conversions()
    if files:
        # 파일 전체 경로를 반환 (드롭다운에는 파일명만 표시)
        choices = files
        labels = [str(Path(f).name) for f in files]
        return gr.update(choices=list(zip(labels, files)), value=files[0] if files else None)
    return gr.update(choices=[], value=None)


# Custom CSS
custom_css = """
.file-container {
    height: 800px;
    overflow-y: auto;
    border: 2px solid #ddd;
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 5px;
    scroll-behavior: smooth;
}

.markdown-container {
    height: 800px;
    overflow-y: auto;
    border: 2px solid #ddd;
    padding: 20px;
    background-color: white;
    border-radius: 5px;
    scroll-behavior: smooth;
}

.file-page {
    margin-bottom: 30px;
    padding-bottom: 20px;
    border-bottom: 2px solid #eee;
}

.page-number {
    background-color: #007acc;
    color: white;
    padding: 5px 15px;
    border-radius: 15px;
    font-size: 14px;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
    display: inline-block;
}

.gradio-container {
    max-width: 1600px !important;
}

/* 스크롤바 커스터마이징 */
.file-container::-webkit-scrollbar,
.markdown-container::-webkit-scrollbar {
    width: 12px;
}

.file-container::-webkit-scrollbar-track,
.markdown-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 6px;
}

.file-container::-webkit-scrollbar-thumb,
.markdown-container::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 6px;
}

.file-container::-webkit-scrollbar-thumb:hover,
.markdown-container::-webkit-scrollbar-thumb:hover {
    background: #555;
}
"""

# Gradio 인터페이스
with gr.Blocks(title="LlamaParse 문서 뷰어", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
        # 📄 LlamaParse 다중 형식 문서 뷰어
        
        **다양한 문서 형식을 LlamaParse로 변환하여 원본과 Markdown을 함께 표시합니다.**
        
        **지원 형식**: PDF, Word(.docx), PowerPoint(.pptx), Excel(.xlsx), 이미지, HWP, 웹 문서(.html) 등
        """
    )
    
    with gr.Tabs():
        with gr.Tab("📤 새로운 변환"):
            with gr.Row():
                file_input = gr.File(
                    label="문서 파일 업로드 (LlamaParse 지원 형식)",
                    file_types=[
                        # PDF
                        ".pdf",
                        # Word Documents
                        ".doc", ".docx", ".docm", ".dot", ".dotm", ".rtf",
                        # PowerPoint
                        ".ppt", ".pptx", ".pptm", ".pot", ".potm", ".potx",
                        # Excel/Spreadsheets
                        ".xlsx", ".xls", ".xlsm", ".xlsb", 
                        # Text files  
                        ".txt", ".xml", ".md", ".json", ".csv",
                        # Web formats
                        ".html", ".htm",
                        # Images
                        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".tiff", ".webp",
                        # eBooks
                        ".epub",
                        # Apple formats
                        ".pages", ".key", ".numbers",
                        # Other formats
                        ".hwp", ".odt", ".ods", ".odp"
                    ],
                    type="filepath"
                )
                convert_btn = gr.Button("🔄 변환하기", variant="primary")
            
            status_text = gr.Textbox(
                label="상태",
                value="지원하는 형식의 문서 파일을 선택한 후 '변환하기' 버튼을 클릭하세요.",
                interactive=False
            )
        
        with gr.Tab("📥 이전 변환 불러오기"):
            with gr.Row():
                with gr.Column():
                    saved_files = gr.Dropdown(
                        label="저장된 변환 파일 (다양한 형식 + Markdown)",
                        choices=[],
                        interactive=True
                    )
                    refresh_btn = gr.Button("🔄 목록 새로고침")
                    load_btn = gr.Button("📂 불러오기 (원본 + Markdown)", variant="primary")
            
            load_status = gr.Textbox(
                label="상태",
                value="불러올 파일을 선택하세요. 원본 문서와 Markdown을 함께 불러옵니다.",
                interactive=False
            )
        
        with gr.Tab("📝 Markdown만 불러오기"):
            with gr.Row():
                with gr.Column():
                    md_file_input = gr.File(
                        label="Markdown 파일 선택",
                        file_types=[".md"],
                        type="filepath"
                    )
                    load_md_btn = gr.Button("📄 Markdown 불러오기", variant="primary")
            
            md_load_status = gr.Textbox(
                label="상태",
                value="Markdown 파일을 선택한 후 '불러오기' 버튼을 클릭하세요. 메인 뷰어에 페이지별로 표시됩니다.",
                interactive=False
            )
    
    gr.Markdown("---")
    gr.Markdown("### 📚 문서와 Markdown 스크롤 뷰어")
    gr.Markdown("*좌우 패널을 각각 스크롤하여 원본 문서와 변환된 내용을 확인하세요.*")
    
    # 원본 문서와 Markdown 표시 영역
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📖 원본 문서")
            file_display = gr.HTML(
                label="원본 문서 전체 페이지",
                elem_classes=["file-container"]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📝 변환된 Markdown")
            markdown_display = gr.Markdown(
                label="Markdown 전체 내용",
                elem_classes=["markdown-container"]
            )
    
    # 이벤트 연결
    
    # 새로운 변환 - 버튼 클릭시에만 변환 실행
    convert_btn.click(
        fn=process_file,
        inputs=[file_input],
        outputs=[file_display, markdown_display, status_text]
    )
    
    # 저장된 변환 불러오기
    refresh_btn.click(
        fn=get_saved_files,
        outputs=[saved_files]
    )
    
    load_btn.click(
        fn=lambda x: load_saved_conversion(x) if x else ("파일을 선택하세요.", "", "파일을 선택하세요."),
        inputs=[saved_files],
        outputs=[file_display, markdown_display, load_status]
    )
    
    # Markdown 파일만 불러오기
    load_md_btn.click(
        fn=load_markdown_file,
        inputs=[md_file_input],
        outputs=[file_display, markdown_display, md_load_status]
    )
    
    # 초기 파일 목록 로드
    demo.load(
        fn=get_saved_files,
        outputs=[saved_files]
    )
    
    gr.Markdown(
        """
        ---
        ### 💡 사용 방법:
        1. **새로운 변환**: 지원하는 문서 파일을 선택한 후 '변환하기' 버튼을 클릭하세요.
        2. **이전 변환 불러오기**: 저장된 MD 파일을 선택하면 원본 문서와 Markdown을 함께 불러옵니다.
        3. **Markdown만 불러오기**: 기존 Markdown 파일을 불러와서 확인할 수 있습니다.
        4. **스크롤 뷰어**: 좌우 패널을 각각 스크롤하여 전체 내용을 확인하세요.
        5. **연속 표시**: 모든 문서 페이지와 Markdown 내용이 연속으로 표시됩니다.
        
        ### 📁 저장 위치:
        - 변환 결과는 `conversions/` 폴더에 `.md` 파일로 저장됩니다.
        - 동일한 파일명이 있으면 자동으로 숫자가 추가됩니다 (예: `document_1.md`, `document_2.md`)
        - MD 파일에는 원본 파일 경로 정보와 페이지별 내용이 포함됩니다.
        
        ### 🔧 특징:
        - **다양한 파일 형식 지원**: PDF, Word, PowerPoint, Excel, 이미지, HWP 등
        - **형식별 최적화**: 파일 형식에 따라 최적화된 parsing instruction 적용
        - **스마트 페이지 표시**: 페이지별로 구분선과 페이지 번호가 표시됩니다
        - **독립 스크롤**: 각 패널을 독립적으로 스크롤할 수 있습니다
        """
    )

if __name__ == "__main__":
    demo.launch(
        share=False,
        # server_name="0.0.0.0",
        server_port=8000,
        show_error=True
    )