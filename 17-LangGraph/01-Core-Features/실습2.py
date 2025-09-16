import gradio as gr
from typing import List, TypedDict, Annotated, Sequence
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
import os
from dotenv import load_dotenv

# .env 파일에서 API 키 로드
load_dotenv()

# -------------------------
# 1. 도구(Tool) 정의
# -------------------------

# Naive RAG를 위한 간단한 문서 검색 도구
docs = {
    "LangChain": "LangChain은 LLM을 체인 형태로 연결해주는 프레임워크입니다.",
    "LangGraph": "LangGraph는 상태 기반의 순환형 LLM 워크플로우를 만들 수 있는 라이브러리입니다.",
    "RAG": "RAG(Retrieval-Augmented Generation)는 외부 지식을 검색하여 LLM의 답변을 보강하는 기술입니다.",
}
vectorstore = Chroma.from_texts(
    list(docs.values()), embedding=OpenAIEmbeddings(), metadatas=[{"source": k} for k in docs.keys()]
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

@tool
def search_docs(query: str) -> str:
    """
    'LangChain', 'LangGraph', 'RAG' 관련 정보를 문서에서 검색합니다.
    사용자가 해당 주제에 대해 질문할 때 이 도구를 사용하세요.
    """
    retrieved_docs = retriever.invoke(query)
    if retrieved_docs:
        return f"[문서 검색 결과]\n- 출처: {retrieved_docs[0].metadata['source']}\n- 내용: {retrieved_docs[0].page_content}"
    return "관련 문서를 찾지 못했습니다."

# -------------------------
# 2. 그래프 상태(State) 정의
# -------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # `interrupt_before`를 위해 추가된 필드
    force_ask_human: bool

# -------------------------
# 3. 모델 및 그래프 노드 정의
# -------------------------

# 모델 정의 (gpt-4o-mini 사용)
llm = ChatOpenAI(model="gpt-4o-mini")
tools = [search_docs]
model_with_tools = llm.bind_tools(tools)

# Agent 노드: 모델을 호출하여 응답 생성
def agent_node(state: AgentState):
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# Tool 노드: 도구 실행 결과를 처리
tool_node = ToolNode(tools)

# 요약 노드: 대화 기록이 길어지면 요약
def summary_node(state: AgentState):
    # 여기서는 간단히 요약 메시지를 추가하는 것으로 대체
    return {"messages": [HumanMessage(content="[이전 대화 내용이 요약되었습니다.]")]}

# -------------------------
# 4. 조건부 엣지(Edge) 정의
# -------------------------

# 도구 사용 여부, 요약 여부, 긴급 상황 여부를 결정
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    # 1. 도구 호출이 있으면 tool_node로
    if last_message.tool_calls:
        return "call_tool"
    
    # 2. 대화 기록이 6개를 넘으면 summarize로
    if len(state["messages"]) > 6:
        return "summarize"
        
    # 3. "긴급" 키워드가 있으면 "emergency_branch"로 (조건부 분기)
    if any("긴급" in m.content for m in state["messages"] if isinstance(m, HumanMessage)):
         return "emergency_branch"

    # 4. 사용자가 강제로 의견을 물었으면 human_in_the_loop로
    if state.get('force_ask_human', False):
        return "human_in_the_loop"

    # 그 외에는 종료
    return END

# 긴급 상황 처리 노드
def emergency_node(state: AgentState):
    return {"messages": [AIMessage(content="🚨 긴급 상황 감지! 관련 부서에 알림을 전송했습니다.")]}

# -------------------------
# 5. 그래프 생성 및 컴파일
# -------------------------

# 메인 그래프
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("summarize", summary_node)
workflow.add_node("emergency_branch", emergency_node)

workflow.add_conditional_edges(
    START, 
    lambda state: "agent", # 시작 시 무조건 agent 노드 호출
)

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "call_tool": "tool_node",
        "summarize": "summarize",
        "emergency_branch": "emergency_branch",
        "human_in_the_loop": END, # 의견 묻기 시 일단 종료 후 사용자 입력 대기
        END: END,
    },
)
workflow.add_edge("tool_node", "agent")
workflow.add_edge("summarize", "agent")
workflow.add_edge("emergency_branch", END)

# 사용자 의견을 묻기 위해 특정 노드 실행 전 인터럽트
# 여기서는 agent 노드 실행 전에 인터럽트를 걸어 사용자의 개입을 허용
app = workflow.compile(
    checkpointer=SqliteSaver.from_conn_string(":memory:"),
    interrupt_before=["agent"],
)

# 병렬 실행(fan-out/fan-in)을 보여주기 위한 별도 그래프
parallel_workflow = StateGraph(AgentState)

def node_a(state):
    last_message = state['messages'][-1].content
    result = f"A: {last_message.upper()}"
    return {"messages": [AIMessage(content=result)]}

def node_b(state):
    last_message = state['messages'][-1].content
    result = f"B: {last_message[::-1]}"
    return {"messages": [AIMessage(content=result)]}

# fan-in을 위한 join 노드
def join_node(state):
    # 병렬 실행된 노드들의 결과를 합침
    combined_message = "\n".join([m.content for m in state['messages'][-2:]])
    final_result = f"\n[A와 B의 결과가 통합됨]\n{combined_message}"
    return {"messages": [AIMessage(content=final_result)]}

parallel_workflow.add_node("A", node_a)
parallel_workflow.add_node("B", node_b)
parallel_workflow.add_node("join", join_node)
parallel_workflow.add_edge(START, "A")
parallel_workflow.add_edge(START, "B")
parallel_workflow.add_edge("A", "join")
parallel_workflow.add_edge("B", "join")
parallel_workflow.add_edge("join", END)
parallel_app = parallel_workflow.compile()


# -------------------------
# 6. Gradio 인터페이스
# -------------------------

def get_graph_image(graph):
    try:
        img_bytes = graph.get_graph().draw_mermaid_png()
        return img_bytes
    except Exception as e:
        print(f"Graph visualization failed: {e}")
        return None

def run_agent(inputs, thread_id, force_ask_human=False):
    if not inputs:
        return [], None, None

    config = {"configurable": {"thread_id": thread_id}}
    
    # 인터럽트 재개
    if app.get_state(config).next:
        response_stream = app.stream(None, config=config)
        # 스트림에서 마지막 AIMessage만 가져옴
        final_response = None
        for chunk in response_stream:
            if messages := chunk.get("agent", {}).get("messages"):
                if isinstance(messages[-1], AIMessage):
                    final_response = messages[-1]
        
        history = app.get_state(config).values['messages']
        chat_interface = [(m.content, "user" if isinstance(m, HumanMessage) else "assistant") for m in history]
        return chat_interface, final_response.content if final_response else "", get_graph_image(app)

    # 새 입력 처리
    message = HumanMessage(content=inputs)
    state_update = {"messages": [message], "force_ask_human": force_ask_human}
    
    response_stream = app.stream(state_update, config=config)
    
    final_response = None
    for chunk in response_stream:
        if messages := chunk.get("agent", {}).get("messages"):
            if isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
                 final_response = messages[-1]

    history = app.get_state(config).values['messages']
    chat_interface = [(m.content, "user" if isinstance(m, HumanMessage) else "assistant") for m in history]
    
    # 인터럽트 발생 시 (의견 묻기)
    if app.get_state(config).next:
        final_response_content = "🤔 당신의 의견은 무엇인가요? 계속 진행하려면 입력을 비우고 전송하세요."
    else:
        final_response_content = final_response.content if final_response else ""

    return chat_interface, final_response_content, get_graph_image(app)


def run_parallel(inputs):
    if not inputs:
        return "", None
    response = parallel_app.invoke({"messages": [HumanMessage(content=inputs)]})
    return response['messages'][-1].content, get_graph_image(parallel_app)


with gr.Blocks(theme=gr.themes.Soft(), title="LangGraph 기능 데모") as demo:
    gr.Markdown("# 🚀 LangGraph 핵심 기능 데모 (gpt-4o-mini)")
    gr.Markdown("`Read.md`의 요구사항을 바탕으로 구현된 Gradio 앱입니다.")

    thread_id_state = gr.State("thread-1") # 세션 유지를 위한 스레드 ID

    with gr.Tab("🤖 Agent & RAG & Summary & Branching"):
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="대화", height=500)
                user_input = gr.Textbox(label="입력")
                with gr.Row():
                    send_btn = gr.Button("전송")
                    ask_opinion_btn = gr.Button("의견 묻기 (Human-in-the-loop)")
            with gr.Column(scale=1):
                graph_output = gr.Image(label="실행 그래프", height=550)
        
        agent_response_output = gr.Textbox(label="최종 응답", interactive=False)

        def agent_chat(msg, tid):
            chat_history, final_response, graph_img = run_agent(msg, tid)
            return chat_history, final_response, graph_img

        def ask_human_chat(msg, tid):
            # 인터럽트 발생 후, 빈 입력으로 재개
            if app.get_state({"configurable": {"thread_id": tid}}).next:
                 chat_history, final_response, graph_img = run_agent(None, tid)
            else: # 새 입력으로 인터럽트 요청
                 chat_history, final_response, graph_img = run_agent(msg, tid, force_ask_human=True)
            return chat_history, final_response, graph_img

        send_btn.click(agent_chat, inputs=[user_input, thread_id_state], outputs=[chatbot, agent_response_output, graph_output])
        ask_opinion_btn.click(ask_human_chat, inputs=[user_input, thread_id_state], outputs=[chatbot, agent_response_output, graph_output])
        
        gr.Examples(
            examples=[
                ["LangGraph에 대해 알려줘"],
                ["RAG가 뭐야?"],
                ["LangChain에 대한 문서를 찾아줘"],
                ["안녕, 넌 누구니?"],
                ["긴급 상황 발생! 즉시 보고해줘"],
                ["이건 긴급한 사안이야"],
                ["오늘 날씨 어때? (도구 미지원)"],
            ],
            inputs=user_input,
            label="예시 질문"
        )

    with gr.Tab("⛓️ Parallel Execution (fan-out/fan-in)"):
        with gr.Row():
            with gr.Column():
                parallel_input = gr.Textbox(label="입력")
                parallel_btn = gr.Button("실행")
                parallel_output = gr.Textbox(label="결과", lines=5)
            with gr.Column():
                parallel_graph = gr.Image(label="병렬 실행 그래프")

        parallel_btn.click(run_parallel, inputs=[parallel_input], outputs=[parallel_output, parallel_graph])
        gr.Examples(
            examples=[
                ["Hello Gradio and LangGraph!"],
                ["This is a parallel execution test."],
                ["동해물과 백두산이 마르고 닳도록"],
            ],
            inputs=parallel_input,
            label="예시 입력"
        )

demo.launch()
