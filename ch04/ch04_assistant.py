##### 기본 정보 입력 ####
# Streamlit 패키지 추가
import streamlit as st

# OpenAI 패키지 추가
from openai import OpenAI

# 채팅 시간을 기록하기 위한 패키지
from datetime import datetime

# 음성 녹음을 관리하기 위한 패키지
from audiorecorder import audiorecorder

# 파이썬 기본 패키지
import os
import numpy as np
import base64

##### 기능 구현 함수 #####
def STT(audio, client):
    # Whisper 모델이 파일 형태로 입력을 받으므로 input.mp3 파일이란 이름으로 음성 파일을 저장합니다.
    filename='input.mp3'
    wav_file = open(filename, "wb")
    wav_file.write(audio.export().read())
    wav_file.close()

    # 음성 파일 열기
    audio_file = open(filename, "rb")
    # Whisper 모델을 활용해 텍스트 얻기
    try:

        # openai 의 whisper API 를 활용하여 텍스트를 추출합니다.
        transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
        )

        # Whisper로 TTS가 끝났으니 이제 mp3 파일을 다시 삭제합니다.
        audio_file.close()
        os.remove(filename)
    except:
        transcript = '여러분들의 Key 값'
    return transcript

def TTS(response):    
    # TTS를 활용하여 만든 음성을 파일로 저장.
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="onyx",
        input=response,
    ) as response:
        filename = "output.mp3"
        response.stream_to_file(filename)

    # 저장한 음성 파일 자동 재생
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        
        # TTS를 통해 생성된 사람 목소리의 음원파일을 재생을 하려면
        # streamlit 의 audio 메서드를 사용해야 합니다.
        # 하지만 audio 메서드는 재생 버튼을 클릭해야만 재생이 됩니다.
        # 따라서 우리가 질문을 하면 따로 답변을 듣는 버튼 클릭 없이 생성이 완료되면
        # 자동 재생될 수 있도록 복잡한 코드로 구현하였습니다.
        # HTML 문법을 사용하여 자동으로 음원을 재생하는 코드를 작성하였고
        # streamlit 안에서 HTML 문법 구현에 사용되는 st.markdown() 을 활용하여 실행을 합니다.
        
        md = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True,)
    # 폴더에 남지 않도록 파일 삭제
    os.remove(filename)

# ChatGPT가 답변을 작성
def ask_gpt(prompt, client):
    response = client.chat.completions.create(model='gpt-3.5-turbo', messages=prompt)
    return response.choices[0].message.content

# 위에서 언급한 함수 STT(Whipser), TTS, ChatGPT를 이용하여 음성 비서 코드를 완성합니다.
st.set_page_config(
    page_title="음성 비서 프로그램🔊",
    layout="wide")

# session state 3개 초기화
# st.session_state["chat"] : 사용자와 음성비서의 대화 내용을 저장하여 채팅창 시각화에 사용
if "chat" not in st.session_state:
    st.session_state["chat"] = []

# st.session_state["check_audio"] : 프로그램이 재실행 될 때마다 이전 녹음파일 정보가 버퍼에
# 남아있어 실행되는 것을 방지하기 위해 이전 녹음파일 정보를 저장합니다
if "check_audio" not in st.session_state:
    st.session_state["check_audio"] = []

# st.session_state["messages"] : GPT API에 입력으로 들어갈 프롬프트 양식. 이전 질문 및 답변을 누적하여 저장.
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": 'You are a thoughtful assistant. Respond to all input in 25 words and answer in korean'}]

# 제목
st.image('ai.png', width=200)
st.header('나만의 인공지능 비서 🔊')
# 구분선
st.markdown('---')
st.subheader('모르는 질문을 하면 답변해줄거에요.🎤')


# OpenAI API 키 지정하기    
client = OpenAI(
        api_key = "여러분들의 Key 값"
)
# 음성 입력 확인 Flag
flag_start = False

# 기능 구현 공간
col1, col2 = st.columns(2)
with col1:
    # 왼쪽 공간 작성
    # 음성 녹음 아이콘 추가
    audio = audiorecorder("질문", "녹음중...")
    if len(audio) > 0 and not np.array_equal(audio, st.session_state["check_audio"]):
        # 음성 재생
        st.audio(audio.export().read())

        # 음원 파일에서 텍스트 추출
        question = STT(audio, client)

        # 채팅 시각화를 위한 질문 내용 저장
        now = datetime.now().strftime("%H:%M")
        st.session_state["chat"] = st.session_state["chat"]+ [("user", now, question)]
        # GPT 모델에 넣을 프롬프트를 위해 질문 저장. 이때 기존 내용 누적.
        st.session_state["messages"] = st.session_state["messages"]+ [{"role": "user", "content": question}]
        # audio 버퍼 확인을 위해 현 시점 오디오 정보 저장
        st.session_state["check_audio"] = audio
        flag_start=True

with col2:
    # 오른쪽 공간 작성
    st.subheader('대화기록 ⌨')
    if flag_start:

        # ChatGPT에게 답변 얻기
        response = ask_gpt(st.session_state["messages"], client)

        # GPT 모델에 넣을 프롬프트를 위해 답변 내용 저장
        st.session_state["messages"] = st.session_state["messages"]+ [{"role": "assistant", "content": response}]

        # 채팅 시각화를 위한 답변 내용 저장
        now = datetime.now().strftime("%H:%M")
        st.session_state["chat"] = st.session_state["chat"]+ [("bot",now, response)]

        # 채팅 형식으로 시각화 하기
        for sender, time, message in st.session_state["chat"]:
            if sender == "user":
                st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                st.write("")
            else:
                st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                st.write("")
        
        # TTS 를 활용하여 음성 파일 생성 및 재생
        TTS(response)