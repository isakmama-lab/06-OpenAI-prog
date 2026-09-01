##### 기본 패키지 #####
# pip install streamlit-audiorecorder
# winget install Gyan.FFmpeg

import streamlit as st
from openai import OpenAI
from datetime import datetime
from audiorecorder import audiorecorder

import os
import numpy as np
import base64


##### STT : Speech → Text #####

def STT(audio, client):

    filename = "input.mp3"

    # 녹음된 음성을 MP3 파일로 저장
    with open(filename, "wb") as f:
        f.write(audio.export(format="mp3").read())

    try:
        # 저장한 음성 파일 열기
        with open(filename, "rb") as audio_file:

            # 음성을 텍스트로 변환
            transcript = client.audio.transcriptions.create(
                model="gpt-transcribe",
                file=audio_file
            )

        return transcript.text

    except Exception as e:
        st.error(f"음성 인식 오류: {e}")
        return ""

    finally:
        # 임시 음성 파일 삭제
        if os.path.exists(filename):
            os.remove(filename)


##### GPT : Responses API #####

def ask_gpt(question, client, previous_response_id=None):

    try:
        # 이전 대화가 없는 첫 질문
        if previous_response_id is None:

            response = client.responses.create(
                model="gpt-5.6-luna",

                instructions=(
                    "당신은 친절한 인공지능 비서입니다. "
                    "사용자의 질문에 한국어로 답변하세요. "
                    "답변은 25단어 이내로 작성하세요."
                ),

                input=question
            )

        # 이전 대화가 있는 경우
        else:

            response = client.responses.create(
                model="gpt-5.6-luna",

                previous_response_id=previous_response_id,

                input=question
            )

        # 생성된 텍스트와 response ID 반환
        return response.output_text, response.id

    except Exception as e:
        st.error(f"GPT 응답 생성 오류: {e}")
        return "", previous_response_id


##### TTS : Text → Speech #####

def TTS(text, client):

    filename = "output.mp3"

    try:
        # 텍스트를 음성으로 변환
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice="onyx",
            input=text,
        ) as response:

            response.stream_to_file(filename)

        # 생성된 MP3 파일 읽기
        with open(filename, "rb") as f:
            data = f.read()

        # Base64 인코딩
        b64 = base64.b64encode(data).decode()

        # 자동 재생용 HTML
        md = f"""
        <audio autoplay="true">
            <source
                src="data:audio/mp3;base64,{b64}"
                type="audio/mp3">
        </audio>
        """

        st.markdown(
            md,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"음성 생성 오류: {e}")

    finally:
        # 임시 음성 파일 삭제
        if os.path.exists(filename):
            os.remove(filename)


##### Streamlit 화면 설정 #####

st.set_page_config(
    page_title="음성 비서 프로그램 🔊",
    layout="wide"
)


##### Session State 초기화 #####

# 화면에 표시할 대화 기록
if "chat" not in st.session_state:
    st.session_state["chat"] = []


# 이전 녹음이 반복 처리되는 것을 방지
if "check_audio" not in st.session_state:
    st.session_state["check_audio"] = []


# Responses API의 이전 응답 ID
if "previous_response_id" not in st.session_state:
    st.session_state["previous_response_id"] = None


##### 화면 구성 #####

st.image(
    "ai.png",
    width=200
)

st.header(
    "나만의 인공지능 비서 🔊"
)

st.markdown("---")

st.subheader(
    "궁금한 내용을 음성으로 질문해 보세요. 🎤"
)


##### OpenAI Client 생성 #####

client = OpenAI(
    api_key="여러분들의 Key 값"
)


##### 새로운 음성 입력 여부 #####

flag_start = False
question = ""


##### 화면을 두 개의 컬럼으로 분리 #####

col1, col2 = st.columns(2)


##### 왼쪽 : 음성 입력 #####

with col1:

    st.subheader("음성 질문 🎤")

    audio = audiorecorder(
        "질문",
        "녹음중..."
    )

    if (
        len(audio) > 0
        and not np.array_equal(
            audio,
            st.session_state["check_audio"]
        )
    ):

        # 사용자가 녹음한 음성 재생
        st.audio(
            audio.export(format="mp3").read()
        )

        # Speech → Text
        question = STT(
            audio,
            client
        )

        if question:

            # 현재 시간
            now = datetime.now().strftime("%H:%M")

            # 화면 출력용 대화 기록 저장
            st.session_state["chat"].append(
                (
                    "user",
                    now,
                    question
                )
            )

            # 현재 오디오를 저장하여 중복 실행 방지
            st.session_state["check_audio"] = audio

            flag_start = True


##### 오른쪽 : AI 답변 #####

with col2:

    st.subheader("대화기록 ⌨")

    if flag_start:

        # Responses API 호출
        answer, response_id = ask_gpt(
            question,
            client,
            st.session_state["previous_response_id"]
        )

        if answer:

            # 다음 질문에서 사용할 Response ID 저장
            st.session_state["previous_response_id"] = response_id

            # 현재 시간
            now = datetime.now().strftime("%H:%M")

            # 화면 출력용 대화 기록 저장
            st.session_state["chat"].append(
                (
                    "bot",
                    now,
                    answer
                )
            )


    ##### 전체 대화 기록 출력 #####

    for sender, time, message in st.session_state["chat"]:

        if sender == "user":

            st.write(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    margin-bottom:10px;
                ">

                    <div style="
                        background-color:#007AFF;
                        color:white;
                        border-radius:12px;
                        padding:8px 12px;
                        margin-right:8px;
                    ">
                        {message}
                    </div>

                    <div style="
                        font-size:0.8rem;
                        color:gray;
                    ">
                        {time}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.write(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:flex-end;
                    margin-bottom:10px;
                ">

                    <div style="
                        font-size:0.8rem;
                        color:gray;
                    ">
                        {time}
                    </div>

                    <div style="
                        background-color:lightgray;
                        border-radius:12px;
                        padding:8px 12px;
                        margin-left:8px;
                    ">
                        {message}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    ##### 새 답변이 만들어졌다면 TTS 실행 #####

    if flag_start and answer:

        TTS(
            answer,
            client
        )