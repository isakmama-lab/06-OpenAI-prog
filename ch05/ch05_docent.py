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
    model="gpt-4-turbo",
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
    max_tokens=1024,
    )
    return response.choices[0].message.content

# TTS
# def TTS(response):    
#     # TTS를 활용하여 만든 음성을 파일로 저장.
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="onyx",
#         input=response
#     )
#     filename = "output.mp3"
#     response.stream_to_file(filename)

#     # 저장한 음성 파일 자동 재생
#     with open(filename, "rb") as f:
#         data = f.read()
#         b64 = base64.b64encode(data).decode()
#         # HTML 문법을 사용하여 자동으로 음원을 재생하는 코드를 작성하여
#         # streamlit 안에서 HTML 문법 구현에 사용되는 st.markdown() 을 활용하여 실행을 합니다.
#         md = f"""
#             <audio autoplay="True">
#             <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
#             </audio>
#             """
#         st.markdown(md, unsafe_allow_html=True,)
#     # 폴더에 남지 않도록 파일 삭제
#     os.remove(filename)

# TTS API를 사용하여 텍스트를 음성으로 변환합니다.
# with_streaming_response.create()를 사용하여
# 생성되는 음성 데이터를 스트리밍 방식으로 파일에 저장합니다.
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

##### 메인 함수 #####
# def main():
#     st.image('ai.png', width=200)
#     st.title("💬 이미지를 해설해드립니다.")

#     # 이미지를 업로드
#     img_file_buffer = st.file_uploader('Upload a PNG image', type='png')

#     if img_file_buffer is not None:
#         image = Image.open(img_file_buffer)

#         # 업로드한 이미지를 화면에 출력
#         st.image(image, caption='Uploaded Image.', width="stretch")
#         # st.image(image, caption='Uploaded Image.', use_column_width=True)

#         # 이미지 => 바이트 버퍼로 변환
#         buffered = io.BytesIO()
#         image.save(buffered, format="PNG")
        
#         # 바이트 버퍼 => Base64 인코딩 바이트 문자열로 변환
#         img_base64 = base64.b64encode(buffered.getvalue())
        
#         # Base64 인코딩 바이트 문자열 => UTF-8 문자열로 디코딩
#         img_base64_str = img_base64.decode('utf-8')

#         # GPT-4V에서 입력받을 수 있는 형태로 변환
#         # 예시 참고: https://platform.openai.com/docs/guides/vision/uploading-base-64-encoded-images
#         image = f"data:image/jpeg;base64,{img_base64_str}"

#         # GPT4V가 이미지에 대한 설명을 반환하고 이를 st.info()로 출력.
#         text = describe(image)
#         st.info(text)

#         # 이미지에 대한 설명을 음성으로 변환.
#         TTS(text)

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
