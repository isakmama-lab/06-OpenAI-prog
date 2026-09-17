##### 기본 정보 입력 ####
import streamlit as st
from openai import OpenAI

import os
import io
import base64
from PIL import Image
from dotenv import load_dotenv

# .env 파일의 환경변수를 불러옵니다.
load_dotenv()

# 환경 변수에서 API KEY를 가져옵니다.
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")

# OpenAI 클라이언트 생성
client = OpenAI(api_key=api_key)


#### 기능 구현 함수 ####
def describe(image_data):
    """Responses API를 사용하여 이미지를 자세히 설명합니다."""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "이 이미지에 대해서 아주 자세히 묘사해줘.",
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data,
                        "detail": "high",
                    },
                ],
            }
        ],
        max_output_tokens=1024,
    )

    # Responses API에서는 생성된 텍스트를 output_text로 간단히 가져올 수 있습니다.
    text = response.output_text

    # TTS에 빈 문자열이 전달되지 않도록 검사합니다.
    if not text or not text.strip():
        raise ValueError("이미지 설명 결과가 비어 있습니다.")

    return text.strip()


def TTS(text):
    """생성된 이미지 설명을 음성으로 변환합니다."""

    # 빈 문자열을 TTS API에 전달하면 400 오류가 발생하므로 사전에 검사합니다.
    if not text or not text.strip():
        st.warning("음성으로 변환할 텍스트가 없습니다.")
        return

    filename = "output.mp3"

    try:
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="onyx",
            input=text,
        ) as response:
            response.stream_to_file(filename)

        # 생성된 MP3 파일을 Base64로 변환하여 브라우저에서 자동 재생합니다.
        # MP3 같은 바이너리 파일을 읽어 → Base64 문자열로 변환하여 → HTML 안에 직접 넣기 위한 과정
        with open(filename, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode("utf-8")

        md = f"""
        <audio autoplay="true" controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """

        st.markdown(md, unsafe_allow_html=True)

    finally:
        # 임시 음성 파일 삭제
        if os.path.exists(filename):
            os.remove(filename)


def main():
    # ai.png가 존재할 때만 화면에 표시합니다.
    if os.path.exists("ai.png"):
        st.image("ai.png", width=200)

    st.title("💬 이미지를 해설해드립니다.")

    # 이미지 업로드
    img_file_buffer = st.file_uploader(
        "PNG 이미지를 업로드하세요.",
        type=["png"],
    )

    if img_file_buffer is not None:
        try:
            # 업로드한 파일을 PIL 이미지 객체로 변환
            pil_image = Image.open(img_file_buffer)

            # 업로드한 이미지를 Streamlit 화면에 출력
            st.image(
                pil_image,
                caption="업로드한 이미지",
                width="stretch",
            )

            # PIL 이미지를 메모리의 바이트 데이터로 변환
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")

            # 이미지 바이트를 Base64 문자열로 인코딩
            img_base64_str = base64.b64encode(
                buffered.getvalue()
            ).decode("utf-8")

            # Responses API의 input_image에서 사용할 Data URL 생성
            image_data = f"data:image/png;base64,{img_base64_str}"

            # GPT에게 이미지 설명 요청
            with st.spinner("이미지를 분석하고 있습니다..."):
                text = describe(image_data)

            # 이미지 설명 출력
            st.subheader("📝 이미지 해설")
            st.info(text)

            # 이미지 설명을 음성으로 변환
            TTS(text)

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    main()
