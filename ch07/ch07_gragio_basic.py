import gradio as gr

def respond(message, history):
    # message: 사용자가 방금 입력한 텍스트
    # history: 이전 대화 내역 [(user_msg, bot_msg), ...]

    # 챗봇 응답 로직 (이 부분에 LLM API 호출 등을 연결할 수 있습니다)
    bot_message = f"답변: '{message}'라고 말씀하셨군요!"

    return bot_message

# ChatInterface를 통한 간단한 챗봇 UI 생성
demo = gr.ChatInterface(
    fn=respond,
    title="기본 AI 챗봇",
    description="Gradio로 구축한 간단한 챗봇 인터페이스입니다.",
    examples=["안녕하세요!", "오늘 날씨 어때요?", "Gradio 사용법을 알려주세요."]
)

if __name__ == "__main__":
    demo.launch()