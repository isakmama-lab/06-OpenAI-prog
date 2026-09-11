# ============================================================
# 0. 라이브러리 불러오기
# ============================================================

from pathlib import Path      # 파일/폴더 경로를 객체 형태로 다룸
import os                    # 환경변수 접근
import re                    # YouTube URL 정규표현식 검사

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pytubefix import YouTube

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. 환경 설정
# ============================================================

load_dotenv()

# 소스코드에 API Key를 직접 작성하지 않고 환경변수에서 읽습니다.
# 이렇게 하면 GitHub 등에 코드를 업로드할 때 API Key 노출 위험을 줄일 수 있습니다.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API Key가 없다면 이후 API 호출이 모두 실패하므로
# 프로그램 시작 시점에 즉시 확인합니다.
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY가 설정되어 있지 않습니다. "
        ".env 파일에 OPENAI_API_KEY 값을 등록하세요."
    )

# OpenAI 공식 SDK 클라이언트
# transcribe_audio()에서 STT API 호출에 사용합니다.
client = OpenAI(api_key=OPENAI_API_KEY)

# 영어 요약 및 한국어 요약에 사용할 언어 모델
# 수업용 예제에서는 비용과 처리 속도를 고려한 경량 모델을 사용합니다.
LLM_MODEL = "gpt-5.6-luna"

# 음성 → 텍스트 변환에 사용할 STT 전용 모델
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"

# 긴 전사문을 나눌 때 하나의 청크가 가질 최대 문자 수
CHUNK_SIZE = 5_000

# 인접한 청크 사이에 300자를 겹치게 하여
# 청크 경계에서 문맥이 끊기는 문제를 줄입니다.
CHUNK_OVERLAP = 300


# ============================================================
# 2. YouTube URL 검사
# ============================================================

def is_valid_youtube_url(url: str) -> bool:

    # YouTube video ID는 일반적으로 11자리입니다.
    pattern = (
        r"^(https?://)?(www\.)?"
        r"(youtube\.com/watch\?v=|youtu\.be/)"
        r"[A-Za-z0-9_-]{11}"
    )

    # re.match()는 패턴이 일치하면 Match 객체,
    # 일치하지 않으면 None을 반환합니다.
    return re.match(pattern, url.strip()) is not None


# ============================================================
# 3. YouTube 오디오 다운로드
# ============================================================

def download_audio(url: str, output_dir: str = "temp") -> Path:

    # 문자열 경로를 Path 객체로 바꿉니다.
    output_path = Path(output_dir)

    # temp 폴더가 없으면 생성합니다.
    output_path.mkdir(parents=True, exist_ok=True)

    # YouTube 영상 정보를 가져옵니다.
    youtube = YouTube(url)

    # 영상 스트림 중 오디오 전용 스트림을 선택합니다.
    audio_stream = youtube.streams.filter(only_audio=True).first()

    if audio_stream is None:
        raise RuntimeError("사용 가능한 오디오 스트림을 찾지 못했습니다.")

    # 실제 오디오 파일을 다운로드합니다.
    downloaded_file = audio_stream.download(output_path=output_path)

    # 문자열이 아닌 Path 객체로 반환하면
    # 이후 파일 열기/삭제 등의 처리가 편리합니다.
    return Path(downloaded_file)


# ============================================================
# 4. OpenAI STT
# ============================================================

def transcribe_audio(file_path: Path) -> str:

    # API 전송을 위해 오디오 파일을 바이너리 읽기 모드로 엽니다.
    with file_path.open("rb") as audio_file:

        transcript = client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio_file,

            # 전사 결과를 문자열 형태로 반환받습니다.
            response_format="text",
        )

    return transcript


# ============================================================
# 5. LangChain LLM 생성
# ============================================================

def create_llm() -> ChatOpenAI:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    # 결과는 Document 객체가 아니라 문자열 리스트입니다.
    return splitter.split_text(transcript)


# ============================================================
# 7. Map → Reduce 방식의 긴 문서 요약
# ============================================================

def summarize_transcript(transcript: str) -> str:

    llm = create_llm()

    # --------------------------------------------------------
    # 7-1. Map Prompt
    # --------------------------------------------------------
    # 각 청크 하나를 요약하기 위한 프롬프트입니다.
    # 실행 시 {text}에 각 chunk 문자열이 들어갑니다.
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

    # --------------------------------------------------------
    # 7-2. Reduce Prompt
    # --------------------------------------------------------
    # 여러 부분 요약을 하나의 최종 영어 요약으로 합칩니다.
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

    # --------------------------------------------------------
    # 7-3. LCEL Chain 구성
    # --------------------------------------------------------
    #
    # LCEL 처리 흐름
    #
    # 입력
    #  ↓
    # Prompt
    #  ↓
    # ChatOpenAI
    #  ↓
    # StrOutputParser
    #  ↓
    # 문자열 결과
    #
    # "|" 연산자로 각 실행 단계를 연결합니다.
    map_chain = map_prompt | llm | StrOutputParser()
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # 전체 전사문을 여러 문자열 청크로 분할합니다.
    chunks = split_transcript(transcript)

    # --------------------------------------------------------
    # 7-4. Map 단계
    # --------------------------------------------------------
    #
    # chunks:
    # ["텍스트1", "텍스트2", "텍스트3"]
    #
    # ↓ 변환
    #
    # [
    #     {"text": "텍스트1"},
    #     {"text": "텍스트2"},
    #     {"text": "텍스트3"},
    # ]
    #
    # batch()를 사용하면 동일한 체인을 여러 입력에 적용할 수 있습니다.
    partial_summaries = map_chain.batch(
        [{"text": chunk} for chunk in chunks]
    )

    # 부분 요약들을 Reduce 단계에 전달하기 위해 하나의 문자열로 합칩니다.
    combined_text = "\n\n".join(partial_summaries)

    # --------------------------------------------------------
    # 7-5. Reduce 단계
    # --------------------------------------------------------
    # 여러 부분 요약을 하나의 최종 영어 요약으로 통합합니다.
    final_summary = reduce_chain.invoke(
        {"text": combined_text}
    )

    return final_summary


# ============================================================
# 8. 한국어 불렛 포인트 요약
# ============================================================

def summarize_in_korean(english_summary: str) -> str:

    llm = create_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 전문 영한 번역가이자 요약가입니다.

입력된 영어 요약을 자연스러운 한국어로 정리하세요.

규칙:
- 반드시 불렛 포인트 형식으로 작성
- 핵심 내용을 빠뜨리지 말 것
- 중복 표현은 제거할 것
- 원문에 없는 내용을 추가하지 말 것
""",
            ),
            ("human", "{text}"),
        ]
    )

    # prompt → model → parser 연결
    chain = prompt | llm | StrOutputParser()

    # invoke()는 하나의 입력을 체인에 전달해 실행합니다.
    return chain.invoke({"text": english_summary})


# ============================================================
# 9. Streamlit UI
# ============================================================

def main() -> None:

    # 페이지 기본 설정
    st.set_page_config(
        page_title="YouTube Summarizer",
        page_icon="📽️",
        layout="wide",
    )

    st.title("📽️ YouTube Summarizer")
    st.caption(
        "YouTube 영상의 음성을 텍스트로 변환한 뒤 "
        "영어 요약과 한국어 핵심 요약을 생성합니다."
    )

    # --------------------------------------------------------
    # 9-1. URL 입력
    # --------------------------------------------------------
    youtube_url = st.text_input(
        "YouTube 주소를 입력하세요.",
        placeholder="https://www.youtube.com/watch?v=***********",
    )

    # URL이 입력되지 않은 상태에서는 아래 처리를 진행하지 않습니다.
    if not youtube_url:
        return

    # YouTube 형식이 아니면 오류 메시지를 출력하고 종료합니다.
    if not is_valid_youtube_url(youtube_url):
        st.error("올바른 YouTube URL을 입력하세요.")
        return

    # 입력한 영상을 Streamlit 화면에서 미리 표시합니다.
    st.video(youtube_url)

    # Streamlit은 위젯을 조작할 때마다 스크립트를 다시 실행하므로
    # 버튼을 눌렀을 때만 비용이 발생하는 AI 작업을 실행합니다.
    if not st.button("영상 요약 시작", type="primary"):
        return

    # finally에서도 사용할 수 있도록 먼저 None으로 초기화합니다.
    audio_file = None

    try:
        # ① YouTube 영상에서 오디오 다운로드
        with st.spinner("YouTube 오디오를 가져오는 중입니다..."):
            audio_file = download_audio(youtube_url)

        # ② 음성을 전체 전사문으로 변환
        with st.spinner("음성을 텍스트로 변환하는 중입니다..."):
            transcript = transcribe_audio(audio_file)

        # 긴 전사문은 기본 화면을 차지하지 않도록 접을 수 있게 표시
        with st.expander("전체 전사문 보기"):
            st.write(transcript)

        # ③ Map → Reduce 방식으로 영어 요약
        with st.spinner("긴 전사문을 요약하는 중입니다..."):
            english_summary = summarize_transcript(transcript)

        st.subheader("English Summary")
        st.success(english_summary)

        # ④ 영어 요약을 한국어 불렛 포인트로 정리
        with st.spinner("한국어 핵심 요약을 생성하는 중입니다..."):
            korean_summary = summarize_in_korean(english_summary)

        st.subheader("한국어 핵심 요약")
        st.info(korean_summary)

    # 네트워크/API/YouTube 다운로드 오류를 사용자 화면에 표시합니다.
    except Exception as error:
        st.error(f"처리 중 오류가 발생했습니다: {error}")

    finally:
        # 성공/실패 여부와 관계없이 다운로드한 임시 파일을 정리합니다.
        if audio_file and audio_file.exists():
            try:
                audio_file.unlink()
            except OSError:
                # 파일 삭제 실패는 핵심 기능 오류가 아니므로 무시합니다.
                pass


# ============================================================
# 10. 프로그램 시작점
# ============================================================

# 이 파일을 직접 실행할 때만 main()을 호출합니다.
#
# 실행 예:
# streamlit run ch08_yt_sum_trans_edu.py
#
# 다른 Python 파일에서 이 모듈을 import하면 main()은 자동 실행되지 않습니다.
if __name__ == "__main__":
    main()
