# config.py
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
    return api_key

# 필요하다면 클라이언트 객체나 키를 변수로 바로 내보낼 수도 있습니다.
API_KEY = os.getenv("OPENAI_API_KEY")