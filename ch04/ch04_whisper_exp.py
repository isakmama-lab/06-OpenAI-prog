from openai import OpenAI	
import os
from dotenv import load_dotenv

# .env파일의 환경변수를 불러옵니다.
load_dotenv()

# 환경 변수에서 API KEY를 가져옵니다.
api_key = os.getenv("OPENAI_API_KEY")
# API 키 입력
client = OpenAI(api_key=api_key)

# 녹음 파일 열기
audio_file = open("speech.mp3", "rb")

# whisper 모델에 음원 파일 넣기
transcript = client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=audio_file, response_format="text")

# 결과 보기
print(transcript)