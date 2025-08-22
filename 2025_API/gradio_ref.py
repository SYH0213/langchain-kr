from operator import itemgetter
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# ChatOpenAI 모델을 초기화합니다.
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 대화형 프롬프트를 생성합니다. 이 프롬프트는 시스템 메시지, 이전 대화 내역, 그리고 사용자 입력을 포함합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 대화 버퍼 메모리를 생성하고, 메시지 반환 기능을 활성화합니다.
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")


class MyConversationChain(Runnable):

    def __init__(self, llm, prompt, memory, input_key="input"):

        self.prompt = prompt
        self.memory = memory
        self.input_key = input_key

        self.chain = (
            RunnablePassthrough.assign(
                chat_history=RunnableLambda(self.memory.load_memory_variables)
                | itemgetter(memory.memory_key)  # memory_key 와 동일하게 입력합니다.
            )
            | prompt
            | llm
            | StrOutputParser()
        )

    def invoke(self, query, configs=None, **kwargs):
        answer = self.chain.invoke({self.input_key: query})
        self.memory.save_context(inputs={"human": query}, outputs={"ai": answer})
        return answer


# ChatOpenAI 모델을 초기화합니다.
llm = ChatOpenAI(model_name="gpt-4.1-mini", temperature=0)

# 대화형 프롬프트를 생성합니다. 이 프롬프트는 시스템 메시지, 이전 대화 내역, 그리고 사용자 입력을 포함합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# 대화 버퍼 메모리를 생성하고, 메시지 반환 기능을 활성화합니다.
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# 요약 메모리로 교체할 경우
# memory = ConversationSummaryMemory(
#     llm=llm, return_messages=True, memory_key="chat_history"
# )

conversation_chain = MyConversationChain(llm, prompt, memory)

# conversation_chain.invoke("안녕하세요? 만나서 반갑습니다. 제 이름은 테디 입니다.")
# conversation_chain.invoke("제 이름이 뭐라고요?")
# conversation_chain.invoke("앞으로는 영어로만 답변해주세요 알겠어요?")
# conversation_chain.invoke("제 이름을 다시 한 번 말해주세요")
# conversation_chain.memory.load_memory_variables({})["chat_history"]
