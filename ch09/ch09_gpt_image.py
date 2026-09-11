from openai import OpenAI
from dotenv import load_dotenv
import base64

# .env 파일의 환경 변수 읽기
load_dotenv()

# OPENAI_API_KEY 환경 변수를 자동으로 사용
client = OpenAI()

response = client.images.generate(
    model="gpt-image-2.5-flare",
    prompt="흰색 배경의 귀엽고 친절한 고객센터 AI 로봇",
    size="1024x1024",
    quality="medium",
)

image_data = base64.b64decode(
    response.data[0].b64_json
)

with open("ai_robot.png", "wb") as f:
    f.write(image_data)

print("ai_robot.png 저장 완료")