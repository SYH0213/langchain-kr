import gradio as gr
import json
import tempfile
import pandas as pd
import tiktoken
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

# --- Core Splitting and Analysis Logic ---
def run_all_analysis(text, splitter_name, chunk_size, chunk_overlap, language):
    # 1. Split Text
    chunks = split_text(text, splitter_name, chunk_size, chunk_overlap, language)
    if not isinstance(chunks, list):
        error_message = chunks
        empty_plot = pd.DataFrame({"Chunk": [], "Length": []})
        return error_message, [], gr.BarPlot(value=empty_plot), None, gr.update(visible=False)

    # 2. Create Visualizations
    boundary_viz = create_boundary_viz(text, chunks)
    length_plot = create_length_plot(chunks)
    
    # 3. Handle Tokenizer Explorer
    is_tokenizer_selected = splitter_name == "TokenTextSplitter"
    tokenizer_output = explore_tokens(text) if is_tokenizer_selected else ""
    tokenizer_visibility = gr.update(visible=is_tokenizer_selected)

    return chunks, boundary_viz, length_plot, tokenizer_output, tokenizer_visibility

def split_text(text, splitter_name, chunk_size, chunk_overlap, language):
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
                "CharacterTextSplitter": CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                "TokenTextSplitter": TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
                "MarkdownHeaderTextSplitter": MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"),("##", "Header 2"),("###", "Header 3")]),
                "HTMLHeaderTextSplitter": HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"),("h2", "Header 2")]),
                "RecursiveJsonSplitter": RecursiveJsonSplitter(max_chunk_size=chunk_size),
            }
            splitter = splitter_map.get(splitter_name)
        if splitter is None: return "잘못된 스플리터를 선택했습니다."
        return splitter.split_text(text)
    except Exception as e:
        return str(e)

def create_boundary_viz(text, chunks):
    viz_data = []
    last_index = 0
    for i, chunk in enumerate(chunks):
        try:
            start_index = text.index(chunk, last_index)
            if start_index > last_index:
                viz_data.append((text[last_index:start_index], None))
            viz_data.append((chunk, f"Chunk {i+1}"))
            last_index = start_index + len(chunk)
        except ValueError:
            viz_data.append((f" [CHUNK {i+1} NOT FOUND IN ORIGINAL TEXT] ", "Error"))
            viz_data.append((chunk, f"Chunk {i+1}"))

    if last_index < len(text):
        viz_data.append((text[last_index:], None))
    return viz_data

def create_length_plot(chunks):
    lengths = [len(chunk) for chunk in chunks]
    df = pd.DataFrame({"Chunk": [f"Chunk {i+1}" for i in range(len(lengths))], "Length": lengths})
    return gr.BarPlot(value=df, x="Chunk", y="Length", title="청크별 길이 (글자 수)", min_width=300)

def explore_tokens(text):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        token_data = [f"[Token {i} | ID: {token_id}]: '{encoding.decode([token_id])}'" for i, token_id in enumerate(tokens)]
        header = f"""총 토큰 수: {len(tokens)}---"""
        return header + "\n" + "\n".join(token_data)
    except Exception as e:
        return f"토큰을 처리하는 중 오류 발생: {e}"

# --- UI Helper Functions ---
def process_file(file):
    if file: return open(file.name, "r", encoding="utf-8").read()
    return ""

def save_session(input_text, output_text):
    if not input_text and not output_text: return None
    session_data = {"input": input_text, "output": json.loads(output_text) if isinstance(output_text, str) else output_text}
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=4)
        return f.name

def load_session(file):
    if file:
        with open(file.name, "r", encoding="utf-8") as f: session_data = json.load(f)
        return session_data.get("input", ""), session_data.get("output", None)
    return "", None

def update_description(splitter_name):
    return SPLITTER_DESCRIPTIONS.get(splitter_name, "")

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 텍스트 분할기 플레이그라운드 v2 (Text Splitter Playground v2)")
    
    with gr.Column():
        gr.Markdown("### 1. 스플리터 설정 (Splitter Settings)")
        splitter_name = gr.Radio(SPLITTER_CHOICES, label="스플리터 선택", value="RecursiveCharacterTextSplitter")
        splitter_description = gr.Markdown(value=SPLITTER_DESCRIPTIONS["RecursiveCharacterTextSplitter"])
        with gr.Row():
            chunk_size = gr.Slider(10, 4000, value=200, step=10, label="청크 사이즈 (Chunk Size)")
            chunk_overlap = gr.Slider(0, 2000, value=50, step=10, label="청크 겹침 (Chunk Overlap)")
        language = gr.Dropdown(["python", "javascript", "typescript", "csharp", "java", "go"], label="언어 (Language for CodeSplitter)", visible=False)

    with gr.Column():
        gr.Markdown("### 2. 텍스트 입력 및 실행 (Input & Actions)")
        input_text = gr.Textbox(lines=15, label="입력 텍스트 (Input Text)")
        with gr.Row():
            file_upload = gr.File(label="텍스트 파일 업로드 (.txt, .md)", file_types=[".txt", ".md"])
            session_load = gr.File(label="세션 불러오기 (.json)", file_types=[".json"])
            session_save = gr.Button("세션 저장 (Save Session)")
        split_button = gr.Button("텍스트 분할 및 분석 실행", variant="primary")
        download_link = gr.File(label="세션 다운로드 (Download Session)", interactive=False)

    with gr.Column():
        gr.Markdown("### 3. 분할 및 분석 결과 (Output & Analysis)")
        with gr.Tabs() as output_tabs:
            with gr.Tab("JSON 결과", id=0):
                output_json = gr.JSON(label="분할된 청크")
            with gr.Tab("경계 시각화", id=1):
                output_boundary_viz = gr.HighlightedText(label="청크 경계 하이라이트", interactive=True)
            with gr.Tab("길이 시각화", id=2):
                output_length_plot = gr.BarPlot(label="청크 길이 그래프")
            with gr.Tab("토크나이저 탐색기", id=3, visible=False) as tokenizer_tab:
                output_tokenizer = gr.Textbox(label="토큰 분해 결과", lines=15, interactive=False)

    # --- Event Handlers ---
    def update_visibility_and_description(splitter):
        is_code = splitter == "CodeSplitter"
        desc = SPLITTER_DESCRIPTIONS.get(splitter, "")
        return gr.update(visible=is_code), desc

    splitter_name.change(update_visibility_and_description, inputs=splitter_name, outputs=[language, splitter_description], show_progress=False)
    splitter_name.change(fn=lambda x: gr.update(visible=x=="TokenTextSplitter"), inputs=splitter_name, outputs=tokenizer_tab, show_progress=False)

    file_upload.upload(process_file, inputs=file_upload, outputs=input_text)
    session_load.upload(load_session, inputs=session_load, outputs=[input_text, output_json])
    session_save.click(save_session, inputs=[input_text, output_json], outputs=download_link)

    split_button.click(
        run_all_analysis,
        inputs=[input_text, splitter_name, chunk_size, chunk_overlap, language],
        outputs=[output_json, output_boundary_viz, output_length_plot, output_tokenizer, tokenizer_tab]
    )

demo.launch()