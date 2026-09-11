"""
ch09_dalle.py
NovelGPT의 이미지 생성을 담당하는 OpenAI Image API 모듈.
"""

import base64
import io

from PIL import Image


def get_image_by_dalle(client, genre: str, img_prompt: str) -> Image.Image:
    """
    이야기 주제와 LLM이 작성한 장면 프롬프트를 이용하여
    GPT Image 모델로 이미지를 생성합니다.

    반환값:
        PIL.Image.Image
    """

    genre = genre.strip()
    img_prompt = img_prompt.strip()

    if not img_prompt:
        raise ValueError("이미지 생성 프롬프트가 비어 있습니다.")

    prompt = f"""
Story title or theme: {genre}

Scene:
{img_prompt}

Create a family-friendly storybook illustration.
Use a polished 3D animated children's movie style,
vibrant colors, expressive characters, cinematic lighting,
and detailed textures.

Do not add text, subtitles, captions, watermarks, or logos.
""".strip()

    response = client.images.generate(
        model="gpt-image-2",
        prompt=prompt,
        size="1024x1024",
        quality="medium",
        n=1,
    )

    if not response.data:
        raise RuntimeError("이미지 생성 결과가 비어 있습니다.")

    image_base64 = response.data[0].b64_json

    if not image_base64:
        raise RuntimeError("이미지 생성 결과에 b64_json 데이터가 없습니다.")

    image_bytes = base64.b64decode(image_base64)

    image = Image.open(io.BytesIO(image_bytes))
    image.load()

    return image
