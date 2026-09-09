import os
import gradio as gr

from openai import OpenAI
from dotenv import load_dotenv


# .env 파일의 환경변수 로드
load_dotenv()

# 환경 변수에서 API Key 가져오기
api_key = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)


def predict(message, history):

    input_messages = []

    # 이전 대화 추가
    for msg in history:

        if isinstance(msg["content"], str):
            text = msg["content"]

        elif isinstance(msg["content"], list):
            text = "".join(
                item.get("text", "")
                for item in msg["content"]
                if isinstance(item, dict)
            )

        else:
            text = str(msg["content"])

        input_messages.append({
            "role": msg["role"],
            "content": text
        })

    # 현재 사용자 질문 추가
    input_messages.append({
        "role": "user",
        "content": message
    })

    # OpenAI Responses API 호출
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="너는 친절하고 유용한 AI 보조원이야.",
        input=input_messages,
        stream=True
    )

    # 스트리밍 응답 누적
    partial_message = ""

    for event in response:

        if event.type == "response.output_text.delta":
            partial_message += event.delta
            yield partial_message


# Gradio ChatInterface 생성
demo = gr.ChatInterface(
    fn=predict,
    title="OpenAI Responses API 기반 Gradio 챗봇",
    description="OpenAI Responses API와 Gradio를 이용한 실시간 스트리밍 챗봇입니다.",
    examples=[
        "파이썬의 주요 특징을 알려줘.",
        "생성형 AI와 머신러닝의 차이를 설명해줘."
    ]
)


# 프로그램 실행
if __name__ == "__main__":
    demo.launch()