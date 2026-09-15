##### 기본 정보 입력 ####
# Streamlit 패키지 추가
import streamlit as st

# OpenAI 패키지 추가
from openai import OpenAI

# 이미지를 처리하기 위한 파이썬 기본 패키지
import os
import io
import base64
from PIL import Image
from dotenv import load_dotenv

# .env파일의 환경변수를 불러옵니다.
load_dotenv()

# 환경 변수에서 API KEY를 가져옵니다.
api_key = os.getenv("OPENAI_API_KEY")

# GPT-4V와 TTS를 위한 client 선언
client = OpenAI(
    api_key = api_key
)

#### 기능 구현 함수 정리 ####
# GPT-4V
def describe(text):
    response = client.chat.completions.create(
    model="gpt-5.6-luna",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": "이 이미지에 대해서 아주 자세히 묘사해줘"},
            {
            "type": "image_url",
            "image_url": {
                "url": text,
            },
            },
        ],
        }
    ],
    max_completion_tokens=1024,
    )
    return response.choices[0].message.content

def TTS(text):
    filename = "output.mp3"

    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="onyx",
            input=text
        ) as response:
            response.stream_to_file(filename)

        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()

        md = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """

        st.markdown(md, unsafe_allow_html=True)

    finally:
        if os.path.exists(filename):
            os.remove(filename)


def main():
    st.image("ai.png", width=200)
    st.title("💬 이미지를 해설해드립니다.")

    # 이미지 업로드
    img_file_buffer = st.file_uploader(
        "Upload a PNG image",
        type="png"
    )

    if img_file_buffer is not None:

        # 업로드한 파일을 PIL 이미지 객체로 변환
        pil_image = Image.open(img_file_buffer)

        # 업로드한 이미지를 Streamlit 화면에 출력
        st.image(
            pil_image,
            caption="Uploaded Image.",
            width="stretch"
        )

        # PIL 이미지를 메모리의 바이트 데이터로 변환
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")

        # 이미지 바이트를 Base64로 인코딩
        img_base64 = base64.b64encode(
            buffered.getvalue()
        )

        # Base64 bytes를 문자열로 변환
        img_base64_str = img_base64.decode("utf-8")

        # OpenAI API에서 사용할 Data URL 생성
        image_data = (
            f"data:image/png;base64,{img_base64_str}"
        )

        # AI에게 이미지 설명 요청
        text = describe(image_data)

        # 이미지 설명을 화면에 출력
        st.info(text)

        # 이미지 설명을 음성으로 변환하여 재생
        TTS(text)

if __name__=="__main__":
    main()
