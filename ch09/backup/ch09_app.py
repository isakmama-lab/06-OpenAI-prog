##### 기본 정보 입력 ####
# Streamlit 패키지 추가
import streamlit as st

# OpenAI 패키지 추가
from openai import OpenAI

# GPT-4와 Dalle를 호출하는 함수
from ch09.backup.ch09_gpt import get_llm # gpt.py로부터 임포트
from ch09.backup.ch09_dalle import get_image_by_dalle # dalle.py로부터 임포트

# 파이썬 기본 패키지
import uuid
import os

st.set_page_config(page_title='📚NovelGPT', layout='wide', initial_sidebar_state='expanded')

##### 기능 구현 함수 정리 #####
# 아래의 함수는 [시작!] 버튼 또는 [진행하기] 버튼을 클릭하면 실행되는 함수.
@st.cache_data(show_spinner='Generating your story...')
def get_output(_pos: st.empty, oid='', genre=''):
    # 아래의 if문은 선택지를 클릭하고 [진행하기] 버튼을 클릭했을 때 동작한다.
    if oid:
	    # 선택지를 클릭하는 순간 직전 과거의 스토리와 선택지의 상태값을 변경.
        st.session_state['genreBox_state'] = True
        st.session_state[f'expanded_{oid}'] = False # 스토리
        st.session_state[f'radio_{oid}_disabled'] = True # 라디오 버튼
        st.session_state[f'submit_{oid}_disabled'] = True # 진행하기 버튼

        # 방금 선택한 선택지에서의 값을 저장.
        user_choice = st.session_state[f'radio_{oid}']
    
    # 처음 시작할 때는 사용자의 선택이 따로 없으므로 user_choice에 사용자가 적은 제목이 저장됨.
    if genre:         
        st.session_state['genreBox_state'] = False
        user_choice = genre
    
    with _pos:
        # 사용자의 선택지로부터 스토리와 이미지를 받아낸다.
        data = get_story_and_image(genre, user_choice)
        add_new_data(data['story'], data['decisionQuestion'], data['choices'], data['dalle_img'])

# 새로운 스토리, 질문, 선택지, 이미지를 반환하는 함수.
def get_story_and_image(genre, user_choice):
    # Dalle 사용을 위해 client 객체를 선언. 이후 get_image_by_dalle()에 전달.
    client = OpenAI()
    # get_llm(): 스토리 전개를 위해 ChatGPT 셋팅하는 함수. 프롬프트도 작성되어져 있음.
    llm_model = get_llm("test")

    # 사용자의 선택지인 user_choice로부터 LLM이 작성한 다음 스토리, 다음 선택지 4개, Dalle 프롬프트를 전달받습니다.
    llm_generation_result = llm_model.invoke({"input": user_choice}, config={"configurable": {"session_id": "test"}}).content

    # 줄바꿈 기준으로 위의 llm_generation_result를 문자열 리스트로 변환. 이렇게 되면 마지막 줄은 Dalle Prompt이다.
    # ex) [스토리 문장1, 스토리 문장2, -- -- --, A선택지, B선택지, C선택지, D선택지, -- -- --, 달리 프롬프트]
    response_list = llm_generation_result.split("\n")
    
    if len(response_list) != 1:
        # 문자열 리스트에서 마지막 원소를 추출하면 달리 프롬프트.
        img_prompt = response_list[-1]
        dalle_img = get_image_by_dalle(client, genre, img_prompt)
    else:
        dalle_img = None
        
    choices = []
    story = ''

    # 메인 스토리(story), 질문(decisionQuestion), 선택지(choices)만 responses의 원소로 남긴다.
    responses = list(filter(lambda x: x != '' and x != '-- -- --', response_list))
    responses = list(filter(lambda x: 'Dalle Prompt' not in x and 'Image prompt' not in x, responses))
    responses = [s for s in responses if s.strip()]
    
    # 메인 스토리(story), 질문(decisionQuestion), 선택지(choices)를 파싱하여 각각 저장.
    for response in responses:
        # 화면에 출력할 선택지 질문에 양 옆에 **를 붙여서 decisionQuestion에 저장.
        # ex) **선택지: 아기 펭귄 보물이는 어떻게 해야할까요?'**
        if response.startswith('선택지:'):
            decisionQuestion = '**' + response + '**'

        elif response[1] == '.':
        # 4개의 선택지를 choices라는 문자열 리스트에 저장
            choices.append(response) 
        # 질문(decisionQuestion)과 선택지(choices)를 제외하면 메인 스토리이므로 story에 저장.
        else:
            story += response + '\n'

    # 스토리에 달리 프롬프트가 여전히 남아있을 경우 제거
    story = story.replace(img_prompt, '')

    return {
        'story': story, # 화면에 출력할 스토리
        'decisionQuestion': decisionQuestion, # 화면에 출력할 질문. '다음은 어떻게 해야할까요?'
        'choices': choices, # 화면에 출력할 실제 4개의 선택지.
        'dalle_img': dalle_img # 화면에 출력할 달리 이미지
    }

# 스토리, 질문, 선택지, 이미지를 저장하는 함수
def add_new_data(*data):
    # uuid.uuid4() 코드를 활용하여 임의의 난수를 생성합니다.
    # ex) oid = fd5198c7-67a5-4fc9-83ad-56afc16e2d6a
    oid = str(uuid.uuid4())

    # 새로운 part의 oid 값을 이전 part의 oid 값들이 저장되어져 있는 리스트에 누적하여 저장합니다.
    st.session_state['oid_list'].append(oid)

    # data_dict에 oid를 key 값으로 현재 part의 데이터를 저장.
    st.session_state['data_dict'][oid] = data
        
# 화면에 각 Part를 출력하는 함수입니다.
def generate_content(story, decisionQuestion, choices: list, img, oid):
    # 과거에 출력된 적이 있던 oid(Part / 스토리는) get_output() 함수의 첫 조건문에서 st.session_state에 기록되었기 때문에 실행되지 않는다.
    if f'expanded_{oid}' not in st.session_state:
        st.session_state[f'expanded_{oid}'] = True # 새로운 스토리를 펼치기 위한 값
    if f'radio_{oid}_disabled' not in st.session_state:
        st.session_state[f'radio_{oid}_disabled'] = False # 4개의 선택지를 선택하는 라디오 버튼을 열기 위한 값
    if f'submit_{oid}_disabled' not in st.session_state:
        st.session_state[f'submit_{oid}_disabled'] = False # 진행하기 버튼을 열기 위한 값
    
    # 화면에 각 스토리 파트가 출력될 때, 'Part 숫자'에서의 숫자를 계산하는 코드입니다. 숫자는 계속 1씩 증가합니다.
    story_pt = list(st.session_state["data_dict"].keys()).index(oid) + 1

    # 각 스토리는 'Part 숫자' 형태로 화면에 출력되며 각 Part는 expanded_{oid}의 값에 따라 열리거나 닫힙니다.
    expander = st.expander(f'Part {story_pt}', expanded=st.session_state[f'expanded_{oid}'])
    col1, col2 = expander.columns([0.65, 0.35])
    empty = st.empty()

    # col2는 스토리 진행 중에 표시될 우측 화면을 의미합니다. 우측 화면에 Dalle가 생성한 이미지를 표현합니다.
    if img:
        col2.image(img, width=40, use_column_width='always')
    
    # col1은 스토리 진행 중에 표시될 좌측 화면을 의미합니다.
    with col1:
        st.write(story)
        
        if decisionQuestion and choices:
            with st.form(key=f'user_choice_{oid}'): 
                st.radio(decisionQuestion, choices, disabled=st.session_state[f'radio_{oid}_disabled'], key=f'radio_{oid}')
                # 진행하기 버튼을 클릭하면 get_output 함수가 실행됩니다.
                # 만약, 이미 진행되었던 Part라면 disabled 값이 True가 되어서 진행하기 버튼을 활성화됩니다.
                st.form_submit_button(
                    label="진행하기", 
                    disabled=st.session_state[f'submit_{oid}_disabled'], 
                    on_click=get_output, args=[empty], kwargs={'oid': oid}
                )

##### 메인 함수 #####
def main():
    # 기본 페이지 설정
    st.title(f"📚 NovelGPT")

    # 스토리 전개 시 각 Part의 데이터를 저장할 리스트.
    if 'data_dict' not in st.session_state:
        st.session_state['data_dict'] = {}

    # 문자열 난수를 저장할 문자열 리스트. 스토리 전개 시 각각의 난수는 각 Part의 Key 값 역할을 하게 된다.
    if 'oid_list' not in st.session_state:
        st.session_state['oid_list'] = []

    # 사용자가 OpenAI API Key 값을 작성하면 저장되는 저장될 변수.
    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = ''

    # 사용자가 OpenAI API Key 값을 작성하는 칸의 활성화 여부. OpenAI Key 값이 입력되기 전에는 칸이 활성화(False) 
    if 'apiBox_state' not in st.session_state:
        st.session_state['apiBox_state'] = False

    # 사용자가 첫 시작 시 주인공 또는 줄거리를 작성하면 저장될 변수. 기본 값은 '아기 펭귄 보물이의 모험'이다.
    if 'genre_input' not in st.session_state:
        st.session_state['genre_input'] = '아기 펭귄 보물이의 모험'
        
    # 사용자가 첫 시작 시 주인공 또는 줄거리를 작성하는 칸의 활성화 여부. OpenAI Key 값이 입력되기 전에는 칸이 비활성화(True) 
    if 'genreBox_state' not in st.session_state:
        st.session_state['genreBox_state'] = True

    # OpenAI API Key를 인증하는 함수입니다.
    def auth():    
        os.environ['OPENAI_API_KEY'] = st.session_state.openai_api_key
        st.session_state.genreBox_state = False

        # API를 입력 칸[ ]의 상태를 반영하는 변수입니다. API Key를 입력(Submit 버튼을 클릭)하면 해당 칸은 비활성화(True).
        st.session_state.apiBox_state = True

    # 좌측의 사이드바 UI
    with st.sidebar:
        st.header('📚 NovelGPT')

        st.markdown('''
        NovelGPT는 소설을 작성하는 인공지능입니다. GPT-4와 Dalle를 사용하여 스토리가 진행됩니다.
        ''')
        
        st.info('**Note:** OpenAI API Key를 입력하세요.')

        # OpenAI Key 값을 입력하는 칸.
        with st.form(key='API Keys'):
            openai_key = st.text_input(
                label='OpenAI API Key', 
                key='openai_api_key',
                type='password', # 입력 시에 값이 화면에 보이지 않고 **로 표시되도록 한다.
                disabled=st.session_state.apiBox_state, # 활성화 여부 변수로 apiBox_state를 사용.
                help='OpenAI API key은 https://platform.openai.com/account/api-keys 에서 발급 가능합니다.',
            )
            
            btn = st.form_submit_button(label='Submit', on_click=auth)

        with st.expander('사용 가이드'):
            st.markdown('''
            - 위의 입력 칸에 <OpenAI API Key>를 작성 후 [Submit] 버튼을 누르세요. 
            - 그 후 우측 화면에 주제나 주인공에 대한 서술을 묘사하고 [시작!] 버튼을 누르세요.
            - 스토리가 시작되면 선택지를 누르며 내용을 전개합니다.
            ''')        

        with st.expander('더 많은 예시 보러가기'):
            st.write('[베스트셀러! 진짜 챗GPT API 활용법](https://www.yes24.com/Product/Goods/121773683)')

    # 시작 시 OpenAI API Key값이 입력되지 않은 경우 경고 문구를 출력합니다.
    if not openai_key.startswith('sk-'): 
        st.warning('OpenAI API Key가 입력되지 않았습니다.', icon='⚠')

    # Genre Input widgets
    with st.container():
        col_1, col_2, col_3 = st.columns([8, 1, 1], gap='small')
        
        col_1.text_input(
            label='Enter the theme/genre of your story',
            key='genre_input',
            placeholder='Enter the theme of which you want the story to be', 
            disabled=st.session_state.genreBox_state
        )
        col_2.write('')
        col_2.write('')
        col_2_cols = col_2.columns([0.5, 6, 0.5])
        col_2_cols[1].button(
            ':arrows_counterclockwise: &nbsp; Clear', 
            key='clear_btn',
            on_click=lambda: setattr(st.session_state, "genre_input", ''),
            disabled=st.session_state.genreBox_state
        )
        col_3.write('')
        col_3.write('')
        # 처음 [시작!] 버튼을 클릭하면 get_output 함수가 실행.
        begin = col_3.button(
            '시작!',
            on_click=get_output, args=[st.empty()], kwargs={'genre': st.session_state.genre_input},
            disabled=st.session_state.genreBox_state
        )

    # 화면에 각 파트를 순서대로 출력합니다.
    # 여기서 각 oid는 각 Part의 key 값 역할을 하는 난수를 의미합니다.
    # oid는 말 그대로 난수이므로 'c4022774-5f2e-4edc-bbbe-cbeed5b98b70' 이런 임의의 값을 가집니다.
    # 모든 oid(Part / 스토리)를 반복문을 통해서 화면에 Part1, Part2 ...와 같이 순차적으로 출력합니다.
    for oid in st.session_state['oid_list']:
        data = st.session_state['data_dict'][oid]
        story = data[0]
        decisionQuestion = data[1]
        chioces = data[2]
        img = data[3]
        # 각 스토리를 출력하는 함수. 이때 지나간 스토리는 화면에서 닫거나 선택지 버튼을 비활성화 하는 등의 역할도 수행.
        generate_content(story, decisionQuestion, chioces, img, oid)

if __name__=="__main__":
    main()