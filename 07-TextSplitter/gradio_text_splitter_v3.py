
import gradio as gr
import json
import tempfile
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    RecursiveJsonSplitter,
    TokenTextSplitter
)

# --- Data ---
SPLITTER_CHOICES = [
    "CharacterTextSplitter", 
    "RecursiveCharacterTextSplitter", 
    "TokenTextSplitter",
    "MarkdownHeaderTextSplitter",
    "HTMLHeaderTextSplitter",
    "CodeSplitter",
    "RecursiveJsonSplitter"
]

SPLITTER_DESCRIPTIONS = {
    "CharacterTextSplitter": "가장 간단한 분할기입니다. 지정된 단일 문자를 기준으로 텍스트를 나눕니다.",
    "RecursiveCharacterTextSplitter": "의미적으로 관련된 텍스트 덩어리를 유지하려고 시도하며, 다양한 구분자 목록을 사용하여 재귀적으로 텍스트를 분할합니다. (권장)",
    "TokenTextSplitter": "토큰을 기준으로 텍스트를 분할합니다. 토큰은 언어 모델이 텍스트를 처리하는 기본 단위입니다.",
    "MarkdownHeaderTextSplitter": "마크다운(#) 헤더를 기준으로 텍스트를 분할하여, 문서의 구조를 유지하는 데 유용합니다.",
    "HTMLHeaderTextSplitter": "HTML (h1, h2 등) 헤더를 기준으로 텍스트를 분할하여, 웹 페이지의 구조를 유지하는 데 유용합니다.",
    "CodeSplitter": "선택한 프로그래밍 언어(Python, JS 등)의 구문을 이해하고, 코드 구조에 맞게 텍스트를 분할합니다.",
    "RecursiveJsonSplitter": "JSON 데이터의 구조를 유지하면서 지정된 크기에 맞게 분할합니다."
}

# --- Core Functions ---
def split_text(text, splitter_name, chunk_size, chunk_overlap, language=None):
    try:
        if splitter_name == "CodeSplitter":
            if language:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            else:
                return "CodeSplitter를 사용하려면 언어를 선택해주세요."
        else:
            splitter_map = {
                "CharacterTextSplitter": CharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                "TokenTextSplitter": TokenTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                "MarkdownHeaderTextSplitter": MarkdownHeaderTextSplitter(
                    headers_to_split_on=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3")]
                ),
                "HTMLHeaderTextSplitter": HTMLHeaderTextSplitter(
                    headers_to_split_on=[("h1", "Header 1"),("h2", "Header 2"),("h3", "Header 3")]
                ),
                "RecursiveJsonSplitter": RecursiveJsonSplitter(max_chunk_size=chunk_size),
            }
            splitter = splitter_map.get(splitter_name)

        if splitter is None:
            return "잘못된 스플리터를 선택했습니다."

        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        return str(e)

# --- UI Helper Functions ---
def get_example_text(splitter_name):
    # ... (same as before)
    return SPLITTER_DESCRIPTIONS.get(splitter_name, "")

def process_file(file):
    if file is not None:
        with open(file.name, "r", encoding="utf-8") as f: return f.read()
    return ""

def save_session_json(input_text, output_text):
    if not input_text and not output_text: return None
    session_data = {"input": input_text, "output": output_text}
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)
        return f.name

def save_session_md(input_text, output_text, chunk_size, chunk_overlap):
    if not any([input_text, output_text, chunk_size, chunk_overlap]): return None
    
    md_content = f"# ✔️ Settings\n\n- **Chunk Size:** `{chunk_size}`\n- **Chunk Overlap:** `{chunk_overlap}`\n\n---\n\n"
    md_content += "## 📜 Input Data\n\n"
    md_content += f"```\n{input_text}\n```\n\n"
    md_content += "\n---\n\n"
    md_content += "## 쪼개진 Output Data\n\n"

    if output_text:
        for i, chunk in enumerate(output_text):
            md_content += f"### Chunk {i+1}\n\n"
            # Check if chunk is a dictionary (from MarkdownHeaderTextSplitter)
            if isinstance(chunk, dict) and 'page_content' in chunk:
                md_content += f"**Metadata:** `{chunk.get('metadata', {})}`\n\n"
                md_content += f"```\n{chunk['page_content']}\n```\n\n"
            else: # Assuming it's a simple string
                md_content += f"```\n{chunk}\n```\n\n"
            md_content += "---\n"


    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.md', encoding='utf-8') as f:
        f.write(md_content)
        return f.name


def load_session(file):
    if file is not None:
        with open(file.name, "r", encoding="utf-8") as f:
            session_data = json.load(f)
            return session_data.get("input", ""), session_data.get("output", None)
    return "", None

def update_description(splitter_name):
    return SPLITTER_DESCRIPTIONS.get(splitter_name, "")

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 텍스트 분할기 플레이그라운드 (Text Splitter Playground) v3")
    
    with gr.Column():
        gr.Markdown("### 1. 스플리터 설정 (Splitter Settings)")
        splitter_name = gr.Radio(
            SPLITTER_CHOICES,
            label="스플리터 선택 (Select Splitter)",
            value="RecursiveCharacterTextSplitter"
        )
        splitter_description = gr.Markdown(value=SPLITTER_DESCRIPTIONS["RecursiveCharacterTextSplitter"], label="스플리터 설명 (Splitter Description)")
        
        with gr.Row():
            chunk_size = gr.Slider(10, 2000, value=200, step=10, label="청크 사이즈 (Chunk Size)")
            chunk_overlap = gr.Slider(0, 1000, value=50, step=10, label="청크 겹침 (Chunk Overlap)")
        
        language = gr.Dropdown(
            ["python", "javascript", "typescript", "csharp", "java", "go"],
            label="언어 (Language for CodeSplitter)",
            visible=False
        )

    with gr.Column():
        gr.Markdown("### 2. 텍스트 입력 및 실행 (Input & Actions)")
        input_text = gr.Textbox(lines=15, label="입력 텍스트 (Input Text)")
        
        with gr.Row():
            file_upload = gr.File(label="텍스트 파일 업로드 (.txt, .md)", file_types=[".txt", ".md"])
            session_load = gr.File(label="세션 불러오기 (.json)", file_types=[".json"])
        
        with gr.Row():
            save_json_button = gr.Button("세션 저장 (.json)")
            save_md_button = gr.Button("세션 저장 (.md)")

        split_button = gr.Button("텍스트 분할 (Split Text)", variant="primary")
        
        with gr.Row():
            download_json = gr.File(label="세션 다운로드 (.json)", interactive=False)
            download_md = gr.File(label="세션 다운로드 (.md)", interactive=False)


    with gr.Column():
        gr.Markdown("### 3. 분할 결과 (Output)")
        output_text = gr.JSON(label="분할된 청크 (Split Chunks)")

    # --- Event Handlers ---
    def update_visibility(splitter):
        return gr.update(visible=splitter == "CodeSplitter")

    splitter_name.change(update_description, inputs=splitter_name, outputs=splitter_description, show_progress=False)
    splitter_name.change(update_visibility, inputs=splitter_name, outputs=language, show_progress=False)
    
    file_upload.upload(process_file, inputs=file_upload, outputs=input_text)
    session_load.upload(load_session, inputs=session_load, outputs=[input_text, output_text])
    
    save_json_button.click(save_session_json, inputs=[input_text, output_text], outputs=download_json)
    save_md_button.click(save_session_md, inputs=[input_text, output_text], outputs=download_md)


    split_button.click(
        split_text, 
        inputs=[input_text, splitter_name, chunk_size, chunk_overlap, language], 
        outputs=output_text
    )

demo.launch()
