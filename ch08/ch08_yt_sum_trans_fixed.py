##### 기본 정보 입력 ####

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pytubefix import YouTube

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI


# ============================================================
# 1. 환경 설정
# ============================================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요."
    )

client = OpenAI(api_key=api_key)

# 요약용 모델
LLM_MODEL = "gpt-5.6-luna"

# 음성 → 텍스트 변환용 모델
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"


# ============================================================
# 2. 기능 구현 함수
# ============================================================

def get_audio(url: str) -> Path:
    """YouTube 영상에서 오디오 스트림을 다운로드합니다."""
    yt = YouTube(url)
    audio = yt.streams.filter(only_audio=True).first()

    if audio is None:
        raise RuntimeError("사용 가능한 오디오 스트림을 찾지 못했습니다.")

    output_dir = Path("temp")
    output_dir.mkdir(exist_ok=True)

    audio_file = audio.download(output_path=output_dir)
    return Path(audio_file)


def get_transcribe(file_path: Path) -> str:
    """오디오 파일을 텍스트로 변환합니다."""
    with file_path.open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            response_format="text",
            file=audio_file,
        )

    return transcript


def create_llm() -> ChatOpenAI:
    """요약/번역에 사용할 LangChain ChatOpenAI 객체를 생성합니다."""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0,
    )


def trans(text: str) -> str:
    """영문 요약을 한국어 불렛 포인트 형태로 번역·요약합니다."""
    llm = create_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 영한 번역가이자 요약가입니다.
입력된 영어 내용을 자연스러운 한국어로 정리하세요.

규칙:
- 반드시 불렛 포인트 형식으로 작성
- 핵심 내용을 빠뜨리지 말 것
- 중복 표현은 제거할 것
- 원문에 없는 내용을 추가하지 말 것""",
            ),
            ("human", "{text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def youtube_url_check(url: str) -> bool:
    # YouTube video ID는 일반적으로 11자리입니다.
    # https://youtu.be/ZZ4B0QUHuNc?si=Nx7xuifeURzHVL5v
    pattern = (
        r"^(https?://)?(www\.)?"
        r"(youtube\.com/watch\?v=|youtu\.be/)"
        r"[A-Za-z0-9_-]{11}"
    )

    # re.match()는 패턴이 일치하면 Match 객체,
    # 일치하지 않으면 None을 반환합니다.
    return re.match(pattern, url.strip()) is not None


def summarize_transcript(transcript: str) -> str:
    """긴 전사문을 Map → Reduce 방식으로 영어 요약합니다."""
    llm = create_llm()

    # Map 단계: 각 청크를 개별 요약
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You summarize YouTube transcripts accurately and concisely.",
            ),
            (
                "human",
                """Summarize the following transcript section in English.
Keep only the important ideas, facts, explanations, and conclusions.

Transcript:
{text}""",
            ),
        ]
    )

    # Reduce 단계: 부분 요약을 하나의 최종 요약으로 통합
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You combine partial summaries into one coherent final summary.",
            ),
            (
                "human",
                """Combine the partial summaries below into one concise English summary.

Requirements:
- About 10 sentences
- Remove repetition
- Preserve the main logical flow
- Do not invent information

Partial summaries:
{text}""",
            ),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    chunks = text_splitter.split_text(transcript)

    partial_summaries = map_chain.batch(
        [{"text": chunk} for chunk in chunks]
    )

    combined_text = "\n\n".join(partial_summaries)

    result = reduce_chain.invoke({"text": combined_text})
    return result


# ============================================================
# 3. Streamlit 메인 함수
# ============================================================

def main():
    st.set_page_config(
        page_title="YouTube Summarize",
        layout="wide",
    )

    st.header("📽️ YouTube Summarizer")

    image_path = Path("ai.png")
    if image_path.exists():
        st.image(str(image_path), width=200)

    youtube_video_url = st.text_input(
        "Please write down the YouTube address. 🖋️",
        placeholder="https://www.youtube.com/watch?v=***********",
    )

    st.markdown("---")

    if not youtube_video_url:
        return

    if not youtube_url_check(youtube_video_url):
        st.error("YouTube URL을 확인하세요.")
        return

    width = 50
    side = width / 2
    _, container, _ = st.columns([side, width, side])
    with container:
        st.video(youtube_video_url)

        # URL 입력만으로 유료 API가 실행되지 않도록 버튼 사용
        if not st.button("영상 요약 시작", type="primary"):
            return

    audio_file = None

    try:
        with st.spinner("YouTube 오디오를 다운로드하는 중입니다..."):
            audio_file = get_audio(youtube_video_url)

        with st.spinner("음성을 텍스트로 변환하는 중입니다..."):
            transcript = get_transcribe(audio_file)

        with st.expander("전체 전사문 보기"):
            st.write(transcript)

        with st.spinner("영어 요약을 생성하는 중입니다..."):
            summary = summarize_transcript(transcript)

        st.subheader("Summary Outcome (in English)")
        st.success(summary)

        with st.spinner("한국어 핵심 요약을 생성하는 중입니다..."):
            korean_summary = trans(summary)

        st.subheader("Final Analysis Result (Reply in Korean)")
        st.info(korean_summary)

    except Exception as error:
        st.error(f"처리 중 오류가 발생했습니다: {error}")

    finally:
        if audio_file and audio_file.exists():
            try:
                audio_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
