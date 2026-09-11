"""
ch09_gpt.py
NovelGPT의 스토리 생성을 담당하는 LangChain 모듈.

- ChatOpenAI 사용
- OpenAI Responses API 사용
- RunnableWithMessageHistory로 대화 문맥 유지
- session_id별 독립적인 대화 기록 관리
"""

from langchain_openai import ChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


# 교육용 메모리 저장소
# 서버가 재시작되면 대화 기록은 초기화됩니다.
_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    session_id별 대화 기록을 반환합니다.

    같은 session_id -> 이전 대화가 이어짐
    다른 session_id -> 서로 독립된 대화
    """
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()

    return _store[session_id]


def clear_session_history(session_id: str) -> None:
    """특정 session_id의 대화 기록을 초기화합니다."""
    history = _store.get(session_id)

    if history is not None:
        history.clear()


def get_llm(api_key: str):
    """
    NovelGPT용 LangChain Runnable을 생성합니다.

    invoke() 호출 시 반드시 session_id를 config로 전달합니다.

    예:
        chain.invoke(
            {"input": "아기 펭귄의 모험"},
            config={
                "configurable": {
                    "session_id": "사용자별 고유 ID"
                }
            }
        )
    """

    # 비용을 낮춘 교육용 모델
    # Responses API 사용을 명시합니다.
    model = ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=api_key,
        use_responses_api=True,
    )

    system_prompt = """
You are NovelGPT, an interactive storybook writer.

Write the story in Korean using polite, natural Korean.

Follow the output format exactly.

[Story rules]
1. Write 2-3 vivid paragraphs for the next part of the story.
2. Then provide exactly four choices.
3. Each choice must begin with A., B., C., or D.
4. Put every choice on its own line.
5. Keep the protagonist's name and important characteristics consistent.
6. Do not choose for the reader.
7. Keep the content family-friendly.
8. Do not refer to yourself in the first person.

[Required output format]

스토리 본문

-- -- --

A. 첫 번째 선택지
B. 두 번째 선택지
C. 세 번째 선택지
D. 네 번째 선택지

-- -- --

선택지: 주인공은 어떻게 해야할까요?

-- -- --

Dalle Prompt Start!
A detailed English image-generation prompt describing the current scene,
characters, setting, lighting, mood, camera composition, and important
visual details.

Important:
- Always include exactly four choices.
- Always include a line beginning with "선택지:".
- Always include "Dalle Prompt Start!".
- Everything after "Dalle Prompt Start!" belongs to the image prompt.
""".strip()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | model

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
