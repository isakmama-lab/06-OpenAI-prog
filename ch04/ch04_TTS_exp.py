from openai	import OpenAI
import os
from dotenv import load_dotenv

# .env파일의 환경변수를 불러옵니다.
load_dotenv()

# 환경 변수에서 API KEY를 가져옵니다.
api_key = os.getenv("OPENAI_API_KEY")

# API 키 입력
client = OpenAI(api_key=api_key)

# 생성할 파일명
speech_file_path = "speech.mp3"

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input="""오늘은 사람들이 좋아하는 것을 만들기에 좋은 날입니다!""",
) as response:
    response.stream_to_file(speech_file_path)