## 내가 이해한것

Agent가 하는일: 개발자가 만들거나 이미 만들어진 tool들의 기능을 Agent가 필요판단하에 tool의 기능을 이용해서 사용자의 요청을 처리함. 이때 tool을 사용하지 않을수도 있고 여러번 재사용 할 수 있음.

코드 작동원리: tool들을 만들거나 가져온것을 리스트에 넣어서 llm, tool, prompt를 create_tool_calling_agent로 엮어서 agent로 만듬. 이때 꼭 create_tool_calling_agent만은 아니고 용도에 따라 다른코드로 agent를 만들기도 함. 만들어진 agent를 AgentExecutor로 엮어서 Agent를 사용함에 있어 설정을 넣어줌. 그 후 langchain을 사용하듯이 invoke, stream등을 사용하면 agent가 tool의 기능을 사용해서 사용자의 요청에 대한 작동을 agent의 판단으로 실행함.

# 📄 오늘 배운 Agent 작동 정리 및 수정사항

## 1. Agent의 역할
- Agent는 **LLM(대규모 언어 모델)**이 사용자의 요청을 처리할 때,  
  필요한 경우 **tool(도구)**을 스스로 선택하고 호출하는 “중간 의사결정자” 역할을 한다.
- LLM은 프롬프트와 도구 목록을 기반으로,  
  **“도구를 쓸지 / 어떤 도구를 쓸지 / 몇 번 쓸지”**를 스스로 판단한다.
- 따라서 Agent는 단순히 도구 모음이 아니라,  
  **LLM의 추론 능력을 활용해 문제 해결 과정을 조율하는 실행 관리자**라 할 수 있다.

---

## 2. 코드 작동 원리
1. **Tool 정의**
   - 직접 만든 함수나 이미 준비된 함수를 `@tool` 데코레이터 또는 `Tool` 객체로 준비한다.
   - 이 Tool들을 리스트에 담는다.  
   ```python
   tools = [add, subtract, multiply, divide]
   ```

2. **Agent 생성**
   - `create_tool_calling_agent` 같은 **생성기 함수**로 LLM, Tool, Prompt를 엮어 Agent를 만든다.
   - 상황에 따라 다른 생성기를 사용할 수도 있다:
     - `create_tool_calling_agent` (가장 일반적)
     - `create_openai_functions_agent` (OpenAI Function Calling API 기반)
     - `create_structured_chat_agent` (JSON 구조 기반)
     - `create_json_agent`, `create_openai_tools_agent` 등

3. **AgentExecutor로 실행 준비**
   - Agent 자체는 “어떻게 도구를 쓸지 결정하는 두뇌”이고,  
     `AgentExecutor`는 그 Agent를 감싸서 실행을 관리하는 래퍼다.
   ```python
   from langchain.agents import AgentExecutor
   executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
   ```

4. **호출**
   - 일반 LLM처럼 `.invoke()`, `.stream()` 등을 사용해 실행한다.
   - 실행 과정에서 LLM은 필요에 따라 tool을 여러 번 호출하거나, 아예 사용하지 않기도 한다.

---

## 3. 작동 흐름 요약

```mermaid
flowchart TD
    A[사용자 입력] --> B[AgentExecutor]
    B --> C[Agent (LLM + Prompt + Tools)]
    C --> D{도구 필요 판단?}
    D -->|예| E[Tool 호출]
    D -->|아니오| F[최종 답변 반환]
    E --> C
    E --> F
```

---

## 4. 핵심 포인트
- **Agent = LLM이 스스로 판단하는 도구 관리자**  
- **Agent 생성기**로 Agent를 만들고 → **AgentExecutor**로 실행한다.  
- **도구 호출은 반복 가능**하며, 실행 과정은 Scratchpad(중간 기록)에 남는다.  
- 프롬프트는 직접 작성하거나 `hub.pull("hwchase17/...")` 같은 템플릿을 활용할 수 있다.  
