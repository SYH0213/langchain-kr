import gradio as gr
from typing import List

# -------------------------
# 1. 사람에게 의견 묻는 노드 (단순 구현)
# -------------------------
def ask_opinion(user_input: str) -> str:
    return f"🤔 제 의견은 이렇습니다: {user_input}에 대해 더 깊게 생각해보면 어떨까요?"

# -------------------------
# 2. 일정 메세지 개수 넘어가면 요약
# -------------------------
chat_history: List[str] = []

def summarize_chat(user_input: str):
    global chat_history
    chat_history.append(f"사용자: {user_input}")
    if len(chat_history) > 6:
        # 앞부분 요약 (여기서는 간단히 문자열로 처리)
        chat_history = ["[이전 대화 요약됨...]"] + chat_history[-2:]
    return "\n".join(chat_history)

# -------------------------
# 3. ToolNode 모의 구현
# -------------------------
def toolnode_demo(query: str):
    # 간단히 "날씨 도구" 예시
    if "날씨" in query:
        return "🌤 오늘은 맑고 기온은 25도입니다."
    return "❌ 해당 질의에 맞는 도구가 없습니다."

# -------------------------
# 4. Agent + ToolNode (간단 흉내)
# -------------------------
def agent_toolnode(query: str):
    if "검색" in query:
        return f"🔎 에이전트가 ToolNode를 호출했습니다: '{query}' 검색 결과입니다."
    return f"🤖 에이전트 응답: '{query}'에 대해 직접 답변합니다."

# -------------------------
# 5. 병렬 실행 (fan-out / fan-in)
# -------------------------
def parallel_nodes(query: str):
    branch_a = f"A 노드 처리 결과: {query.upper()}"
    branch_b = f"B 노드 처리 결과: {query[::-1]}"
    combined = f"{branch_a}\n{branch_b}\n➡ fan-in 결과: {branch_a} + {branch_b}"
    return combined

# -------------------------
# 6. Conditional branching
# -------------------------
def conditional_branch(query: str):
    if "긴급" in query:
        return "🚨 긴급 처리 경로로 이동!"
    else:
        return "✅ 일반 처리 경로로 이동!"

# -------------------------
# 7. Naive RAG (간단 검색)
# -------------------------
docs = {
    "LangChain": "LangChain은 LLM을 체인 형태로 연결해주는 프레임워크입니다.",
    "LangGraph": "LangGraph는 상태 기반 LLM 워크플로우를 만들 수 있습니다.",
}
def naive_rag(query: str):
    for key, value in docs.items():
        if key.lower() in query.lower():
            return f"📄 문서에서 찾음: {value}"
    return "❓ 관련 문서를 찾지 못했습니다."

# -------------------------
# Gradio 인터페이스
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🚀 오늘 배운 기능 실습 Gradio 데모")

    with gr.Tab("1. 의견 묻기"):
        inp = gr.Textbox(label="질문 입력")
        out = gr.Textbox(label="노드 응답")
        inp.submit(ask_opinion, inp, out)

    with gr.Tab("2. 대화 기록 + 요약"):
        inp2 = gr.Textbox(label="메시지 입력")
        out2 = gr.Textbox(label="대화 기록", lines=10)
        inp2.submit(summarize_chat, inp2, out2)

    with gr.Tab("3. ToolNode"):
        inp3 = gr.Textbox(label="쿼리 입력")
        out3 = gr.Textbox(label="도구 응답")
        inp3.submit(toolnode_demo, inp3, out3)

    with gr.Tab("4. Agent + ToolNode"):
        inp4 = gr.Textbox(label="쿼리 입력")
        out4 = gr.Textbox(label="에이전트 응답")
        inp4.submit(agent_toolnode, inp4, out4)

    with gr.Tab("5. 병렬 실행"):
        inp5 = gr.Textbox(label="쿼리 입력")
        out5 = gr.Textbox(label="fan-out/fan-in 결과", lines=5)
        inp5.submit(parallel_nodes, inp5, out5)

    with gr.Tab("6. 조건 분기"):
        inp6 = gr.Textbox(label="쿼리 입력")
        out6 = gr.Textbox(label="분기 결과")
        inp6.submit(conditional_branch, inp6, out6)

    with gr.Tab("7. Naive RAG"):
        inp7 = gr.Textbox(label="쿼리 입력")
        out7 = gr.Textbox(label="검색 결과")
        inp7.submit(naive_rag, inp7, out7)

demo.launch()
