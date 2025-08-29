# -*- coding: utf-8 -*-
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
import numpy as np
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import re
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import warnings
import hashlib
import json
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='gradio.components.dropdown')


# --- 1. 데이터 소스 정의 ---
def get_sample_data():
    """내장 샘플 데이터를 반환합니다."""
    return [
        {"category": "일상", "text": "오늘 날씨가 정말 좋네요."},
        {"category": "일상", "text": "점심으로 무엇을 먹을까요?"},
        {"category": "일상", "text": "저는 지금 커피를 마시고 있어요."},
        {"category": "일상", "text": "What a beautiful day!"},
        {"category": "IT", "text": "최신 AI 모델의 성능이 놀랍습니다."},
        {"category": "IT", "text": "파이썬은 배우기 쉬운 프로그래밍 언어입니다."},
        {"category": "IT", "text": "이 버그는 어떻게 해결해야 할까요?"},
        {"category": "IT", "text": "The new API is much faster."},
        {"category": "비즈니스", "text": "다음 분기 실적 발표가 기대됩니다."},
        {"category": "비즈니스", "text": "새로운 마케팅 전략이 필요합니다."},
        {"category": "비즈니스", "text": "계약서 검토를 완료했습니다."},
        {"category": "비즈니스", "text": "We need to increase our market share."},
        {"category": "과학", "text": "우주의 신비는 끝이 없습니다."},
        {"category": "과학", "text": "새로운 논문이 네이처에 게재되었습니다."},
        {"category": "과학", "text": "양자역학은 매우 흥미로운 분야입니다."},
        {"category": "과학", "text": "Photosynthesis is a complex process."},
        {"category": "스포츠", "text": "손흥민 선수가 또 골을 넣었습니다."},
        {"category": "스포츠", "text": "이번 월드컵은 정말 재미있네요."},
        {"category": "스포츠", "text": "저는 야구 보는 것을 좋아합니다."},
        {"category": "스포츠", "text": "The crowd went wild after the touchdown."},
        {"category": "예술", "text": "이 그림은 정말 아름답습니다."},
        {"category": "예술", "text": "저는 클래식 음악을 즐겨 듣습니다."},
        {"category": "예술", "text": "셰익스피어의 희곡은 시대를 초월합니다."},
        {"category": "예술", "text": "Her voice is simply mesmerizing."},
        {"category": "역사", "text": "조선왕조실록은 유네스코 세계기록유산입니다."},
        {"category": "역사", "text": "로마 제국의 역사는 매우 흥미롭습니다."},
        {"category": "역사", "text": "제2차 세계대전은 인류에게 큰 상처를 남겼습니다."},
        {"category": "역사", "text": "The Renaissance was a pivotal period in history."},
        {"category": "건강", "text": "규칙적인 운동은 건강에 좋습니다."},
        {"category": "건강", "text": "충분한 수면을 취하는 것이 중요합니다."},
        {"category": "건강", "text": "비타민 C는 면역력 강화에 도움이 됩니다."},
        {"category": "건강", "text": "A balanced diet is key to good health."},
    ]

def load_sentences(source_type, file_obj, use_chunking, chunk_size, chunk_overlap):
    """데이터 소스 타입에 따라 문장 리스트를 로드합니다."""
    if source_type == "내장 샘플":
        return get_sample_data()
    if file_obj is None:
        gr.Warning("파일을 업로드해주세요.")
        return []

    file_path = file_obj.name
    _, file_extension = os.path.splitext(file_path)
    
    raw_text = ""
    try:
        if file_extension.lower() == ".pdf":
            doc = fitz.open(file_path)
            for page in doc:
                raw_text += page.get_text("text") + "\n"
            doc.close()
        elif file_extension.lower() == ".csv":
            # CSV는 청킹 대상이 아니므로 기존 로직 유지
            df = pd.read_csv(file_path)
            sentences = []
            for _, row in df.iterrows():
                if 'text_kr' in row and pd.notna(row['text_kr']):
                    sentences.append({"category": row.get('category', 'CSV'), "text": row['text_kr']})
                if 'text_en' in row and pd.notna(row['text_en']):
                    sentences.append({"category": row.get('category', 'CSV'), "text": row['text_en']})
            return sentences
        else:
            gr.Warning(f"지원하지 않는 파일 형식입니다: {file_extension}")
            return []

        # PDF 또는 다른 텍스트 파일에 대한 청킹 처리
        if use_chunking and raw_text and source_type == "PDF 업로드":
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_text(raw_text)
            return [{"category": "PDF-Chunked", "text": chunk} for chunk in chunks if chunk.strip()]
        elif raw_text:
            # 청킹 사용 안 할 경우, 기존처럼 줄바꿈으로 분리
            lines = raw_text.split('\n')
            return [{"category": "PDF", "text": line.strip()} for line in lines if line.strip()]
        else:
            return []

    except Exception as e:
        gr.Error(f"파일 처리 중 오류 발생: {e}")
        return []


# --- 2. 임베딩 모델 로더 ---
def get_embedder(model_name):
    """선택된 모델에 대한 임베딩 함수/객체를 반환합니다."""
    if model_name == "HuggingFace (multilingual-e5-large-instruct)":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

    elif model_name == "OpenAI (text-embedding-3-small)":
        if not os.environ.get("OPENAI_API_KEY"):
            gr.Warning("OPENAI_API_KEY가 설정되지 않았습니다.")
            return None
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    elif model_name == "Upstage (solar-embedding-1-large)":
        if not os.environ.get("UPSTAGE_API_KEY"):
            gr.Warning("UPSTAGE_API_KEY가 설정되지 않았습니다.")
            return None
        from langchain_upstage import UpstageEmbeddings
        # Upstage는 질문/문서 모델이 분리되어 있으므로 튜플로 반환
        query_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")
        passage_embedder = UpstageEmbeddings(model="solar-embedding-1-large-passage")
        return (query_embedder, passage_embedder)

    elif model_name == "Ollama (nomic-embed-text)":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model="nomic-embed-text")

    else:
        gr.Error("알 수 없는 모델입니다.")
        return None

# --- 3. Vector Store 및 유사도 계산 ---

CACHE_DIR = Path("./vs_cache")

def get_config_hash(config):
    """설정 딕셔너리로부터 고유한 해시 값을 생성합니다."""
    # 순서에 상관없이 동일한 해시를 얻기 위해 딕셔너리를 정렬된 JSON 문자열로 변환
    serialized_config = json.dumps(config, sort_keys=True)
    return hashlib.sha1(serialized_config.encode()).hexdigest()

def get_db_path(file_name, model_name, preprocess_options, use_chunking, chunk_size, chunk_overlap):
    """설정 값들을 기반으로 DB 저장 경로를 생성합니다."""
    config = {
        "file_name": file_name,
        "model_name": model_name,
        "preprocess": sorted(preprocess_options), # 순서 보장
        "chunking": {
            "use": use_chunking,
            "size": chunk_size,
            "overlap": chunk_overlap
        },
        "distance_metric": "cosine"  # 코사인 거리 사용을 명시
    }
    config_hash = get_config_hash(config)
    return CACHE_DIR / config_hash

def preprocess_text(text, options):
    """텍스트 전처리 옵션을 적용합니다."""
    if "소문자화" in options:
        text = text.lower()
    if "숫자/기호 제거" in options:
        text = re.sub(r'[^\w\s]', '', text)
    if "중복 공백 정리" in options:
        text = re.sub(r'\s+', ' ', text).strip()
    return text


def calculate_similarity(query, data, model_name, top_k, threshold, preprocess_options):
    """유사도 계산의 메인 로직"""
    if not query:
        gr.Warning("기준 문장을 입력해주세요.")
        return pd.DataFrame(), None, "기준 문장을 입력해주세요."
    if not data:
        gr.Warning("비교할 데이터가 없습니다.")
        return pd.DataFrame(), None, "비교할 데이터가 없습니다."

    try:
        embedder = get_embedder(model_name)
        if embedder is None:
            raise ValueError(f"{model_name} 모델을 불러올 수 없습니다. API 키 등을 확인해주세요.")
    except Exception as e:
        gr.Error(f"임베딩 모델 로딩 실패: {e}")
        return pd.DataFrame(), None, f"임베딩 모델 로딩 실패: {e}"

    processed_query = preprocess_text(query, preprocess_options)
    sentences = [item['text'] for item in data]
    processed_sentences = [preprocess_text(s, preprocess_options) for s in sentences]

    try:
        # Upstage 모델의 경우 (query, passage) 튜플로 반환됨
        if isinstance(embedder, tuple):
            query_embedder, passage_embedder = embedder
            query_vec = query_embedder.embed_query(processed_query)
            doc_vecs = passage_embedder.embed_documents(processed_sentences)
        else: # 그 외 모델
            query_vec = embedder.embed_query(processed_query)
            doc_vecs = embedder.embed_documents(processed_sentences)
    except Exception as e:
        error_message = f"임베딩 생성 중 오류: {e}"
        if "Connection refused" in str(e) and "Ollama" in model_name:
            error_message += "\n\nOllama 모델을 사용하려면 로컬에 Ollama 서비스가 실행 중이어야 합니다."
        gr.Error(error_message)
        return pd.DataFrame(), None, error_message

    query_vec_normalized = normalize([query_vec], norm='l2')[0]
    doc_vecs_normalized = normalize(doc_vecs, norm='l2')
    sim_matrix = cosine_similarity([query_vec_normalized], doc_vecs_normalized)[0]

    results = []
    for i, score in enumerate(sim_matrix):
        if score >= threshold:
            results.append({
                "유사도": score,
                "문장": sentences[i],
                "카테고리": data[i]['category']
            })

    if not results:
        summary_message = f"유사도 임계치({threshold})를 넘는 문장을 찾지 못했습니다. 임계치를 낮추거나 다른 모델을 사용해 보세요."
        return pd.DataFrame(), None, summary_message

    df = pd.DataFrame(results).sort_values(by="유사도", ascending=False).head(top_k)
    if not df.empty:
        df.insert(0, "순위", range(1, len(df) + 1))
        df['유사도'] = df['유사도'].map('{:.4f}'.format)

    fig = create_plot(df)
    summary = create_summary(df, model_name)

    return df, fig, summary

def create_plot(df):
    """결과 데이터프레임을 받아 막대 그래프를 생성합니다."""
    if df.empty:
        return None
    fig, ax = plt.subplots()
    labels = [f"Rank {r}" for r in df['순위']]
    scores = [float(s) for s in df['유사도']]
    ax.barh(labels, scores)
    ax.invert_yaxis()
    ax.set_xlabel('Similarity')
    ax.set_title('Top-K Similarity Scores')
    plt.tight_layout()
    return fig

def create_summary(df, model_name):
    """결과를 바탕으로 간단한 요약 문구를 생성합니다."""
    if df.empty:
        return f"[{model_name}] 모델에 대한 결과가 없습니다."

    top_category = df['카테고리'].iloc[0]
    category_concentration = (df['카테고리'] == top_category).sum() / len(df)

    summary_parts = []
    summary_parts.append(f"[{model_name}] 분석 결과:")
    summary_parts.append(f"- 가장 유사한 문장은 '{df['문장'].iloc[0]}' (유사도: {df['유사도'].iloc[0]}) 입니다.")
    summary_parts.append(f"- 상위 {len(df)}개 중 {category_concentration:.0%}가 '{top_category}' 카테고리에 속해, 모델이 카테고리를 잘 군집화하는 것으로 보입니다.")
    
    return "\n".join(summary_parts)

# --- 4. Gradio UI 정의 ---

# 시작 시 사용 가능한 모델 목록 생성
available_models = ["HuggingFace (multilingual-e5-large-instruct)", "Ollama (nomic-embed-text)"]
if os.environ.get("OPENAI_API_KEY"):
    available_models.append("OpenAI (text-embedding-3-small)")
if os.environ.get("UPSTAGE_API_KEY"):
    available_models.append("Upstage (solar-embedding-1-large)")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Vector Store를 메모리에 저장하기 위한 상태 변수
    vector_store_state = gr.State(None)

    gr.Markdown("# 📖 문장 임베딩 기반 유사도 비교 데모 v2 (VectorStore)")
    gr.Markdown("ChromaDB를 사용하여 임베딩을 캐싱하고 재사용하는 기능이 추가되었습니다.\n" 
                "**참고:** Ollama 모델은 로컬에 [Ollama](https://ollama.com/)가 설치 및 실행되어 있어야 합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## ⬅️ 설정")
            
            with gr.Accordion("1. 데이터 소스 선택", open=True):
                data_source_radio = gr.Radio(
                    ["내장 샘플", "PDF 업로드", "CSV 업로드"],
                    label="데이터 소스",
                    value="내장 샘플"
                )
                file_uploader = gr.File(label="파일 업로드", file_types=[".pdf", ".csv"], visible=False)

            with gr.Accordion("2. 임베딩 및 전처리 설정", open=True):
                model_selector = gr.Dropdown(
                    available_models,
                    label="임베딩 모델 선택",
                    value=available_models[0] if available_models else None,
                    interactive=bool(available_models)
                )
                preprocess_options_checkbox = gr.CheckboxGroup(
                    ["소문자화", "숫자/기호 제거", "중복 공백 정리"],
                    label="전처리 옵션"
                )

            with gr.Accordion("3. 청킹 옵션 (PDF 전용)", open=False) as chunking_accordion:
                use_chunking_checkbox = gr.Checkbox(label="업로드 파일에 청킹 적용", value=True)
                chunk_size_slider = gr.Slider(100, 2000, value=500, step=50, label="청크 크기 (Chunk Size)")
                chunk_overlap_slider = gr.Slider(0, 500, value=50, step=10, label="청크 중첩 (Chunk Overlap)")

            with gr.Accordion("4. 검색 방식 및 파라미터 설정", open=True):
                backend_selector = gr.Radio(
                    ["직접 계산 (기존 방식)", "ChromaDB Vector Store"],
                    label="백엔드 선택",
                    value="ChromaDB Vector Store"
                )
                with gr.Row():
                    top_k_slider = gr.Slider(1, 20, value=10, step=1, label="Top-K")
                    threshold_slider = gr.Slider(0, 1, value=0.2, step=0.05, label="유사도 임계치 (직접 계산 전용)")

            with gr.Accordion("5. Vector Store 관리", open=True):
                vs_status_textbox = gr.Textbox(label="Vector Store 상태", interactive=False)
                create_vs_button = gr.Button("🔄 Vector Store 생성/로드", variant="secondary")

            gr.Markdown("---")
            gr.Markdown("## 🚀 실행")
            query_input = gr.Textbox(lines=3, label="기준 문장", placeholder="여기에 비교할 문장을 입력하세요...")
            run_button = gr.Button("✨ 유사도 계산 실행", variant="primary")
            
            gr.Examples(
                examples=["김치에 대해 알려줘", "마리오에 대해 알려줘", "케이팝 콘서트의 의미는 뭐야?", "전 세계 사람들이 즐기는 문화는 뭐가 있어?"],
                inputs=query_input,
                label="예시 질문"
            )

        with gr.Column(scale=2):
            gr.Markdown("## ➡️ 결과")
            result_table = gr.DataFrame(label="유사도 순위")
            result_plot = gr.Plot(label="유사도 시각화")
            summary_output = gr.Textbox(label="결과 요약", lines=5, interactive=False)


    # --- 5. 이벤트 핸들러 및 실행 로직 ---

    def create_or_load_vector_store(source_type, file_obj, model_name, preprocess_options, use_chunking, chunk_size, chunk_overlap):
        """Vector Store를 생성하거나 로드합니다."""
        if (source_type == "PDF 업로드" or source_type == "CSV 업로드") and file_obj is None:
            gr.Warning(f"{source_type}를 선택했지만 파일이 업로드되지 않았습니다.")
            return None, "오류: 파일이 없습니다."
        
        file_name = os.path.basename(file_obj.name) if file_obj else "내장 샘플"

        db_path = get_db_path(file_name, model_name, preprocess_options, use_chunking, chunk_size, chunk_overlap)

        embedder = get_embedder(model_name)
        if embedder is None:
            gr.Error(f"{model_name} 모델을 불러올 수 없습니다. API 키 등을 확인해주세요.")
            return None, "오류: 임베딩 모델 로딩 실패"

        if db_path.exists():
            gr.Info(f"기존 Vector Store를 로드합니다: {db_path}")
            vector_store = Chroma(persist_directory=str(db_path), embedding_function=embedder)
            status_message = f"✅ 로드 완료: {db_path.name}"
            return vector_store, status_message
        
        gr.Info(f"새로운 Vector Store를 생성합니다: {db_path}")
        data = load_sentences(source_type, file_obj, use_chunking, chunk_size, chunk_overlap)
        if not data:
            gr.Warning("데이터를 불러오지 못했습니다.")
            return None, "오류: 데이터 로딩 실패"

        texts = [preprocess_text(item['text'], preprocess_options) for item in data]
        metadatas = [{'category': item['category']} for item in data]
        documents = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]

        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedder,
                persist_directory=str(db_path),
                collection_metadata={"hnsw:space": "cosine"}
            )
            status_message = f"✅ 생성 및 저장 완료: {db_path.name}"
            return vector_store, status_message
        except Exception as e:
            gr.Error(f"Vector Store 생성 중 오류 발생: {e}")
            return None, f"오류: {e}"

    def search_from_vector_store(query, vector_store, top_k, model_name, preprocess_options):
        """Vector Store에서 유사도 검색을 수행합니다."""
        if not query:
            gr.Warning("기준 문장을 입력해주세요.")
            return pd.DataFrame(), None, "기준 문장을 입력해주세요."
        if vector_store is None:
            gr.Warning("Vector Store가 준비되지 않았습니다. 먼저 Vector Store를 생성/로드해주세요.")
            return pd.DataFrame(), None, "Vector Store가 준비되지 않았습니다."

        processed_query = preprocess_text(query, preprocess_options)
        
        results_with_scores = vector_store.similarity_search_with_score(
            query=processed_query, 
            k=top_k
        )

        if not results_with_scores:
            return pd.DataFrame(), None, "유사한 문장을 찾지 못했습니다."

        results = []
        for doc, score in results_with_scores:
            similarity = 1 - score
            results.append({
                "유사도": similarity,
                "문장": doc.page_content,
                "카테고리": doc.metadata.get('category', 'N/A')
            })
        
        df = pd.DataFrame(results)
        if not df.empty:
            df.insert(0, "순위", range(1, len(df) + 1))
            df['유사도'] = df['유사도'].map('{:.4f}'.format)

        fig = create_plot(df)
        summary = create_summary(df, model_name)

        return df, fig, summary

    def run_analysis_wrapper(backend, query, vector_store, model_name, top_k, threshold, preprocess_options, source_type, file_obj, use_chunking, chunk_size, chunk_overlap):
        """백엔드 선택에 따라 분석을 실행하는 라우터 함수"""
        if backend == "직접 계산 (기존 방식)":
            data = load_sentences(source_type, file_obj, use_chunking, chunk_size, chunk_overlap)
            if not data:
                return pd.DataFrame(), None, "데이터를 불러오지 못했습니다. 소스를 확인해주세요.", vector_store
            df, fig, summary = calculate_similarity(query, data, model_name, top_k, threshold, preprocess_options)
            return df, fig, summary, vector_store
        
        elif backend == "ChromaDB Vector Store":
            df, fig, summary = search_from_vector_store(query, vector_store, top_k, model_name, preprocess_options)
            return df, fig, summary, vector_store
        
        else:
            raise ValueError("알 수 없는 백엔드입니다.")

    # --- 이벤트 리스너 연결 ---
    def on_source_change(source_type):
        is_upload = source_type in ["PDF 업로드", "CSV 업로드"]
        show_chunking = source_type == "PDF 업로드"
        return {
            file_uploader: gr.update(visible=is_upload),
            chunking_accordion: gr.update(visible=show_chunking)
        }

    data_source_radio.change(fn=on_source_change, inputs=data_source_radio, outputs=[file_uploader, chunking_accordion])

    def on_setting_change():
        return None, "⚠️ 설정이 변경되었습니다. Vector Store를 다시 생성/로드해야 합니다."
    
    settings_inputs = [data_source_radio, file_uploader, model_selector, preprocess_options_checkbox, use_chunking_checkbox, chunk_size_slider, chunk_overlap_slider]
    for setting_input in settings_inputs:
        setting_input.change(fn=on_setting_change, outputs=[vector_store_state, vs_status_textbox])

    create_vs_button.click(
        fn=create_or_load_vector_store,
        inputs=[data_source_radio, file_uploader, model_selector, preprocess_options_checkbox, use_chunking_checkbox, chunk_size_slider, chunk_overlap_slider],
        outputs=[vector_store_state, vs_status_textbox]
    )

    run_button.click(
        fn=run_analysis_wrapper,
        inputs=[
            backend_selector, query_input, vector_store_state, model_selector, top_k_slider, threshold_slider, 
            preprocess_options_checkbox, data_source_radio, file_uploader,
            use_chunking_checkbox, chunk_size_slider, chunk_overlap_slider
        ],
        outputs=[result_table, result_plot, summary_output, vector_store_state]
    )


if __name__ == "__main__":
    if not available_models:
        print("오류: 사용 가능한 임베딩 모델이 없습니다. API 키 또는 로컬 모델 설정을 확인해주세요.")
    else:
        print(f"사용 가능한 모델: {available_models}")
    
    demo.launch()

