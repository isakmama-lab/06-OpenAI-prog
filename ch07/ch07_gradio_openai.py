import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


# .env 파일의 환경 변수 로드
load_dotenv()

# API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)


def extract_text(content):
    """content를 문자열로 변환합니다."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                item_text = item.get("text", "")
                if isinstance(item_text, str):
                    text_parts.append(item_text)

        return "".join(text_parts)

    return str(content)


def predict(message, history):
    input_messages = []

    # 이전 대화 추가
    for msg in history:
        # 최신 Gradio 형식: {"role": "...", "content": "..."}
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")

        # 일부 Gradio 버전의 구형 형식: (user_message, assistant_message)
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            user_message, assistant_message = msg

            if user_message is not None:
                input_messages.append({
                    "role": "user",
                    "content": extract_text(user_message),
                })

            if assistant_message is not None:
                input_messages.append({
                    "role": "assistant",
                    "content": extract_text(assistant_message),
                })

            continue

        else:
            continue

        input_messages.append({
            "role": role,
            "content": extract_text(content),
        })

    # 현재 사용자 질문 추가
    input_messages.append({
        "role": "user",
        "content": message,
    })

    # Responses API 스트리밍 호출
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="너는 친절하고 유용한 AI 보조원이야.",
        input=input_messages,
        stream=True,
    )

    # 스트리밍 응답 누적
    partial_message = ""

    for event in response:
        if event.type == "response.output_text.delta":
            partial_message += event.delta
            yield partial_message


demo = gr.ChatInterface(
    fn=predict,
    title="OpenAI Responses API 기반 Gradio 챗봇",
    description="OpenAI Responses API와 Gradio를 이용한 실시간 스트리밍 챗봇입니다.",
    examples=[
        "파이썬의 주요 특징을 알려줘.",
        "생성형 AI와 머신러닝의 차이를 설명해줘.",
    ],
)


if __name__ == "__main__":
    demo.launch()