import gradio as gr
from operator import itemgetter
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationEntityMemory,
    ConversationKGMemory,
    VectorStoreRetrieverMemory,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

# 환경 변수 로드
load_dotenv()



class MyConversationChain(Runnable):
    def __init__(self, llm, prompt, memory, input_key="input"):
        self.prompt = prompt
        self.memory = memory
        self.input_key = input_key

        # 메모리 타입에 따라 적절한 memory_key 설정
        memory_key = self.get_memory_key(memory)

        if isinstance(memory, VectorStoreRetrieverMemory):
            # VectorStoreRetrieverMemory의 경우 검색된 관련 대화를 사용
            def get_vector_history(inputs):
                try:
                    memory_vars = self.memory.load_memory_variables(inputs)
                    history_text = memory_vars.get("history", "")
                    if history_text:
                        # 문자열을 간단한 메시지 형태로 변환
                        from langchain_core.messages import HumanMessage, AIMessage

                        messages = []
                        # 간단한 파싱으로 대화 기록 추출 (실제로는 더 정교한 파싱이 필요할 수 있음)
                        if "Human:" in history_text and "AI:" in history_text:
                            parts = history_text.split("\n")
                            current_human = ""
                            current_ai = ""
                            for part in parts:
                                if part.startswith("Human:"):
                                    if current_human and current_ai:
                                        messages.extend(
                                            [
                                                HumanMessage(content=current_human),
                                                AIMessage(content=current_ai),
                                            ]
                                        )
                                    current_human = part.replace("Human:", "").strip()
                                    current_ai = ""
                                elif part.startswith("AI:"):
                                    current_ai = part.replace("AI:", "").strip()
                                elif current_ai:
                                    current_ai += " " + part.strip()
                                elif current_human:
                                    current_human += " " + part.strip()
                            if current_human and current_ai:
                                messages.extend(
                                    [
                                        HumanMessage(content=current_human),
                                        AIMessage(content=current_ai),
                                    ]
                                )
                        return messages
                    return []
                except:
                    return []

            self.chain = (
                RunnablePassthrough.assign(
                    chat_history=RunnableLambda(get_vector_history)
                )
                | prompt
                | llm
                | StrOutputParser()
            )
        else:
            self.chain = (
                RunnablePassthrough.assign(
                    chat_history=RunnableLambda(self.memory.load_memory_variables)
                    | itemgetter(memory_key)
                )
                | prompt
                | llm
                | StrOutputParser()
            )

    def get_memory_key(self, memory):
        """메모리 타입에 따라 올바른 memory_key 반환"""
        if isinstance(memory, VectorStoreRetrieverMemory):
            return "history"
        elif hasattr(memory, "memory_key"):
            return memory.memory_key
        elif hasattr(memory, "chat_memory_key"):
            return memory.chat_memory_key
        else:
            # 기본값으로 'history' 사용 (대부분의 메모리 타입에서 사용)
            return "history"

    def invoke(self, query, configs=None, **kwargs):
        answer = self.chain.invoke({self.input_key: query})
        self.memory.save_context(inputs={"human": query}, outputs={"ai": answer})
        return answer

    def clear_memory(self):
        self.memory.clear()


class GradioChatApp:
    def __init__(
        self,
        model_name="gpt-4o-mini",
        temperature=0.0,
        memory_type="ConversationBufferMemory",
        memory_params=None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.memory_type = memory_type
        self.memory_params = memory_params or {}
        self.conversation_chain = None
        self.initialize_chain()

    def initialize_chain(self):
        # ChatOpenAI 모델 초기화
        llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)

        # 대화형 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful and friendly AI assistant. Respond in Korean unless otherwise requested.",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        # 선택한 메모리 타입으로 메모리 생성
        memory = self.create_memory(self.memory_type, **self.memory_params)

        # 대화 체인 생성
        self.conversation_chain = MyConversationChain(llm, prompt, memory)

    def create_memory(self, memory_type, **params):
        """선택한 메모리 타입으로 메모리 생성"""
        if memory_type == "ConversationBufferMemory":
            return ConversationBufferMemory(
                return_messages=True, memory_key="chat_history"
            )

        elif memory_type == "ConversationBufferWindowMemory":
            k = params.get("k", 2)
            return ConversationBufferWindowMemory(
                k=k, return_messages=True, memory_key="chat_history"
            )

        elif memory_type == "ConversationTokenBufferMemory":
            llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
            max_token_limit = params.get("max_token_limit", 100)
            return ConversationTokenBufferMemory(
                llm=llm,
                max_token_limit=max_token_limit,
                return_messages=True,
                memory_key="chat_history",
            )

        elif memory_type == "ConversationEntityMemory":
            llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
            # ConversationEntityMemory는 memory_key 파라미터를 지원하지 않을 수 있음
            try:
                return ConversationEntityMemory(
                    llm=llm, return_messages=True, memory_key="chat_history"
                )
            except TypeError:
                return ConversationEntityMemory(llm=llm, return_messages=True)

        elif memory_type == "ConversationKnowledgeGraph":
            llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
            # ConversationKGMemory는 memory_key 파라미터를 지원하지 않을 수 있음
            try:
                return ConversationKGMemory(
                    llm=llm, return_messages=True, memory_key="chat_history"
                )
            except TypeError:
                return ConversationKGMemory(llm=llm, return_messages=True)

        elif memory_type == "ConversationSummary":
            llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
            max_token_limit = params.get("max_token_limit", 100)
            return ConversationSummaryMemory(
                llm=llm,
                max_token_limit=max_token_limit,
                return_messages=True,
                memory_key="chat_history",
            )

        elif memory_type == "VectorStoreRetrieverMemory":
            # 임베딩 모델을 정의합니다.
            embeddings_model = OpenAIEmbeddings()

            # Vector Store 를 초기화 합니다.
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)
            vectorstore = FAISS(embeddings_model, index, InMemoryDocstore({}), {})

            # 벡터 조회가 여전히 의미적으로 관련성 있는 정보를 반환한다는 것을 보여주기 위해서입니다.
            k = params.get("k", 1)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            return VectorStoreRetrieverMemory(retriever=retriever)

        else:
            # 기본값
            return ConversationBufferWindowMemory(
                k=2, return_messages=True, memory_key="chat_history"
            )

    def get_vector_search_result(self, query):
        """VectorStoreRetrieverMemory에서 유사한 대화 검색"""
        if self.memory_type == "VectorStoreRetrieverMemory":
            try:
                # 메모리에서 관련 대화 검색
                memory_vars = self.conversation_chain.memory.load_memory_variables(
                    {"input": query}
                )
                history_text = memory_vars.get("history", "")
                if history_text:
                    return history_text
                else:
                    return "데이터 없음"
            except Exception as e:
                return f"검색 오류: {str(e)}"
        return ""

    def respond(self, message, history):
        """Gradio 채팅 인터페이스를 위한 응답 함수"""
        try:
            # LangChain 대화 체인을 통해 응답 생성
            response = self.conversation_chain.invoke(message)
            return response
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

    def clear_conversation(self):
        """대화 기록 초기화"""
        if self.conversation_chain:
            self.conversation_chain.clear_memory()
        return None, None

    def update_settings(self, model_name, temperature, memory_type, memory_params):
        """모델 및 메모리 설정 업데이트"""
        self.model_name = model_name
        self.temperature = temperature
        self.memory_type = memory_type
        self.memory_params = memory_params
        self.initialize_chain()
        return f"설정이 업데이트되었습니다. 메모리 타입: {memory_type}"


def create_gradio_interface():
    """Gradio 인터페이스 생성"""
    app = GradioChatApp()

    with gr.Blocks(title="LangChain Gradio Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 LangChain Gradio Chat
            LangChain과 OpenAI를 활용한 대화형 AI 챗봇입니다.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="대화",
                    height=500,
                    bubble_full_width=False,
                    avatar_images=(None, "🤖"),
                )
                msg = gr.Textbox(
                    label="메시지 입력",
                    placeholder="메시지를 입력하세요... (Enter 키로 전송)",
                    lines=2,
                )

                # VectorStoreRetrieverMemory 검색 결과 표시 창
                with gr.Group() as vector_search_group:
                    vector_search_output = gr.Textbox(
                        label="🔍 벡터 검색 결과 (유사한 대화 기록)",
                        placeholder="VectorStoreRetrieverMemory 사용 시 검색된 관련 대화가 여기에 표시됩니다.",
                        lines=5,
                        interactive=False,
                        visible=False,
                    )

                with gr.Row():
                    submit = gr.Button("전송", variant="primary")
                    clear = gr.Button("대화 초기화")

            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 설정")

                # 모델 설정
                model_dropdown = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    value="gpt-4o-mini",
                    label="모델 선택",
                )
                temperature_slider = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=0.0,
                    step=0.1,
                    label="Temperature (창의성 수준)",
                )

                # 메모리 설정
                gr.Markdown("#### 💭 메모리 설정")
                memory_dropdown = gr.Dropdown(
                    choices=[
                        "ConversationBufferMemory",
                        "ConversationBufferWindowMemory",
                        "ConversationTokenBufferMemory",
                        "ConversationEntityMemory",
                        "ConversationKnowledgeGraph",
                        "ConversationSummary",
                        "VectorStoreRetrieverMemory",
                    ],
                    value="ConversationBufferMemory",
                    label="메모리 타입",
                    info="대화 기록을 저장하는 방식을 선택하세요",
                )

                # 메모리별 설정 파라미터
                with gr.Group(visible=False) as window_params:
                    k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=2,
                        step=1,
                        label="Window Size (k) - 기억할 대화 턴 수",
                    )

                with gr.Group(visible=False) as token_params:
                    token_slider = gr.Slider(
                        minimum=50,
                        maximum=1000,
                        value=100,
                        step=50,
                        label="Max Token Limit - 최대 토큰 수",
                    )

                with gr.Group(visible=False) as summary_params:
                    summary_token_slider = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=100,
                        step=25,
                        label="Max Token Limit - 요약 생성 기준 토큰 수",
                    )

                with gr.Group(visible=False) as vector_params:
                    vector_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=1,
                        step=1,
                        label="Vector Search K - 검색할 벡터 개수",
                    )

                update_btn = gr.Button("설정 적용", variant="secondary")
                settings_output = gr.Textbox(label="설정 상태", interactive=False)

                gr.Markdown(
                    """
                    ### 📝 사용 방법
                    1. 메시지를 입력하고 Enter 또는 전송 버튼을 클릭
                    2. AI가 선택한 메모리 방식으로 대화 맥락을 기억
                    3. 새로운 대화를 시작하려면 '대화 초기화' 클릭
                    4. 설정 변경 시 '설정 적용' 클릭
                    
                    ### 💭 메모리 타입 설명
                    - **Buffer**: 모든 대화 기록 저장
                    - **BufferWindow**: 최근 k개 대화만 기억
                    - **TokenBuffer**: 토큰 수 제한으로 기록 관리
                    - **Entity**: 중요 개체 정보 추출하여 기억
                    - **KnowledgeGraph**: 지식 그래프로 관계 저장
                    - **Summary**: 오래된 대화를 요약하여 저장
                    - **VectorStoreRetriever**: 벡터 유사도 기반 검색으로 관련 대화 기억
                    """
                )

        # 이벤트 핸들러 연결
        def user_submit(user_message, history):
            if not user_message:
                return "", history
            history = history or []
            return "", history + [[user_message, None]]

        def bot_respond(history):
            if history and history[-1][1] is None:
                user_message = history[-1][0]

                # VectorStoreRetrieverMemory 사용 시 검색 결과 가져오기
                vector_search_result = ""
                if app.memory_type == "VectorStoreRetrieverMemory":
                    vector_search_result = app.get_vector_search_result(user_message)

                bot_message = app.respond(user_message, history[:-1])
                history[-1][1] = bot_message

                return history, vector_search_result
            return history, ""

        def update_memory_ui(memory_type):
            """메모리 타입에 따라 설정 UI 표시/숨김"""
            vector_search_visible = memory_type == "VectorStoreRetrieverMemory"

            if memory_type == "ConversationBufferWindowMemory":
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=vector_search_visible),
                )
            elif memory_type == "ConversationTokenBufferMemory":
                return (
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=vector_search_visible),
                )
            elif memory_type == "ConversationSummary":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=vector_search_visible),
                )
            elif memory_type == "VectorStoreRetrieverMemory":
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=vector_search_visible),
                )
            else:
                return (
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=vector_search_visible),
                )

        def apply_settings(
            model_name,
            temperature,
            memory_type,
            k_value,
            token_limit,
            summary_token_limit,
            vector_k,
        ):
            """설정 적용"""
            # 메모리 타입에 따른 파라미터 설정
            memory_params = {}
            if memory_type == "ConversationBufferWindowMemory":
                memory_params["k"] = int(k_value)
            elif memory_type == "ConversationTokenBufferMemory":
                memory_params["max_token_limit"] = int(token_limit)
            elif memory_type == "ConversationSummary":
                memory_params["max_token_limit"] = int(summary_token_limit)
            elif memory_type == "VectorStoreRetrieverMemory":
                memory_params["k"] = int(vector_k)

            # 설정 업데이트
            result = app.update_settings(
                model_name, temperature, memory_type, memory_params
            )
            return result

        # 메시지 전송 이벤트
        msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_respond, chatbot, [chatbot, vector_search_output]
        )
        submit.click(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_respond, chatbot, [chatbot, vector_search_output]
        )

        # 대화 초기화
        def clear_conversation():
            """대화 내역과 메모리 모두 초기화"""
            app.clear_conversation()  # 메모리 초기화
            return [], "", ""  # 채팅봇 화면, 입력창, 벡터 검색 결과 초기화

        clear.click(clear_conversation, outputs=[chatbot, msg, vector_search_output])

        # 메모리 타입 변경시 UI 업데이트
        memory_dropdown.change(
            update_memory_ui,
            inputs=[memory_dropdown],
            outputs=[
                window_params,
                token_params,
                summary_params,
                vector_params,
                vector_search_output,
            ],
        )

        # 설정 업데이트
        update_btn.click(
            apply_settings,
            inputs=[
                model_dropdown,
                temperature_slider,
                memory_dropdown,
                k_slider,
                token_slider,
                summary_token_slider,
                vector_k_slider,
            ],
            outputs=[settings_output],
        )

        # 예제 입력
        gr.Examples(
            examples=[
                "안녕하세요! 자기소개를 해주세요.",
                "제 이름은 김철수이고, 30살 개발자입니다.",
                "방금 제가 말한 제 정보를 기억하고 있나요?",
                "파이썬으로 피보나치 수열을 구현하는 방법을 알려주세요.",
                "이전에 설명한 피보나치 수열과 관련해서 다른 질문이 있어요.",
                "처음 대화 내용을 다시 말해주세요.",
            ],
            inputs=msg,
        )

    return demo


if __name__ == "__main__":
    # Gradio 인터페이스 실행
    demo = create_gradio_interface()
    demo.launch(
        share=False,  # True로 설정하면 공개 URL 생성
        # server_name="0.0.0.0"
        server_port=7860,  # 포트 설정
        show_error=True,
    )
