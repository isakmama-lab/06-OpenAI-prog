import sys
from pathlib import Path

# 상위 폴더(루트 경로)를 파이썬 모듈 검색 경로에 추가
try:
    current_dir = Path(__file__).resolve().parent
except NameError:
    current_dir = Path.cwd()

sys.path.append(str(current_dir.parent))

from openai	import OpenAI
from config import get_api_key


# API_KEY 함수 호출하여 사용
api_key = get_api_key()

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