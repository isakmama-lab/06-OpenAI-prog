import io
from PIL import Image
import base64

# 이미지 프롬프트를 입력받으면 이미지를 생성하여 전달.
def get_image_by_dalle(client, genre, img_prompt):    
    response = client.images.generate(
    model="gpt-image-2.5-flare",
    prompt='The name of this story is ' + genre + ' ' + img_prompt + ' The style is 3D computer-rendered children’s movie animation with vibrant colors and detailed textures.',
    size="1024x1024",
    quality="medium",
    n=1,
    response_format='b64_json'
    )

    image_data = base64.b64decode(response.data[0].b64_json)
    image = Image.open(io.BytesIO(image_data))
    return image