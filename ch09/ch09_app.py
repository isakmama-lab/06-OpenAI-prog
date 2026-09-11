"""
ch09_app.py
Streamlit 기반 NovelGPT 애플리케이션.

최종 실행본 v2 - API Key 입력용 state와 실제 저장 state를 분리.

실행:
    streamlit run ch09_app.py
"""

import re
import uuid

import streamlit as st
from openai import OpenAI

from ch09_gpt import clear_session_history, get_llm
from ch09_dalle import get_image_by_dalle


st.set_page_config(
    page_title="📚 NovelGPT",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 1. Session State 초기화
# =========================================================
def init_session_state():
    """앱에서 사용하는 Streamlit session_state를 초기화합니다."""

    defaults = {
        "data_dict": {},
        "oid_list": [],
        "llm_session_id": str(uuid.uuid4()),
        "api_key_input": "",
        "openai_api_key": "",
        "api_authenticated": False,
        "genre_input": "아기 펭귄 보물이의 모험",
        "story_genre": "",
        "story_started": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =========================================================
# 2. LLM 응답을 문자열로 변환
# =========================================================
def message_to_text(message) -> str:
    """
    LangChain AIMessage의 응답을 문자열로 변환합니다.

    Responses API를 사용하는 경우에도 AIMessage.text를 우선 사용하고,
    환경 차이를 고려해 content도 보조적으로 처리합니다.
    """

    text = getattr(message, "text", None)

    if isinstance(text, str) and text.strip():
        return text.strip()

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                block_text = block.get("text")

                if isinstance(block_text, str):
                    parts.append(block_text)

        return "\n".join(parts).strip()

    return str(content).strip()


# =========================================================
# 3. LLM 응답 파싱
# =========================================================
def parse_story_response(text: str) -> dict:
    """
    LLM 응답을 다음 요소로 분리합니다.

    - story
    - decisionQuestion
    - choices
    - img_prompt

    'Dalle Prompt Start!' 이후의 모든 텍스트를 이미지 프롬프트로
    처리하므로 이미지 프롬프트가 여러 줄이어도 정상 동작합니다.
    """

    if not text:
        raise ValueError("LLM 응답이 비어 있습니다.")

    marker = "Dalle Prompt Start!"

    if marker not in text:
        raise ValueError(
            f"LLM 응답에서 '{marker}'를 찾을 수 없습니다."
        )

    story_section, img_prompt = text.split(marker, 1)
    img_prompt = img_prompt.strip()

    if not img_prompt:
        raise ValueError("LLM이 이미지 프롬프트를 생성하지 않았습니다.")

    story_lines = []
    choices = []
    decision_question = ""

    for raw_line in story_section.splitlines():
        line = raw_line.strip()

        if not line or line == "-- -- --":
            continue

        if line.startswith("선택지:"):
            decision_question = line
            continue

        if re.match(r"^[A-D]\.\s+", line):
            choices.append(line)
            continue

        story_lines.append(line)

    story = "\n\n".join(story_lines).strip()

    if not story:
        raise ValueError("스토리 본문을 찾을 수 없습니다.")

    if len(choices) != 4:
        raise ValueError(
            f"선택지는 정확히 4개여야 하지만 {len(choices)}개가 반환되었습니다."
        )

    if not decision_question:
        raise ValueError("선택 질문('선택지: ...')을 찾을 수 없습니다.")

    return {
        "story": story,
        "decisionQuestion": f"**{decision_question}**",
        "choices": choices,
        "img_prompt": img_prompt,
    }


# =========================================================
# 4. Story + Image 생성
# =========================================================
def get_story_and_image(genre: str, user_input: str) -> dict:
    """
    1) LangChain을 이용하여 다음 스토리를 생성하고
    2) GPT Image 모델을 이용하여 장면 이미지를 생성합니다.
    """

    api_key = st.session_state["openai_api_key"].strip()

    if not api_key:
        raise ValueError("OpenAI API Key가 설정되지 않았습니다.")

    client = OpenAI(api_key=api_key)
    llm = get_llm(api_key)

    result = llm.invoke(
        {"input": user_input},
        config={
            "configurable": {
                "session_id": st.session_state["llm_session_id"]
            }
        },
    )

    llm_text = message_to_text(result)
    parsed = parse_story_response(llm_text)

    image = get_image_by_dalle(
        client=client,
        genre=genre,
        img_prompt=parsed["img_prompt"],
    )

    return {
        "story": parsed["story"],
        "decisionQuestion": parsed["decisionQuestion"],
        "choices": parsed["choices"],
        "dalle_img": image,
    }


# =========================================================
# 5. 생성 데이터 저장
# =========================================================
def add_new_data(story, decision_question, choices, img):
    """새로운 Story Part를 session_state에 저장합니다."""

    oid = str(uuid.uuid4())

    st.session_state["oid_list"].append(oid)

    st.session_state["data_dict"][oid] = {
        "story": story,
        "decisionQuestion": decision_question,
        "choices": choices,
        "img": img,
    }


# =========================================================
# 6. 새로운 Story Part 생성
# =========================================================
def create_next_part(user_input: str):
    """사용자 입력 또는 선택지를 이용하여 다음 Part를 생성합니다."""

    genre = st.session_state["story_genre"]

    with st.spinner("스토리와 이미지를 생성하고 있습니다..."):
        data = get_story_and_image(
            genre=genre,
            user_input=user_input,
        )

    add_new_data(
        story=data["story"],
        decision_question=data["decisionQuestion"],
        choices=data["choices"],
        img=data["dalle_img"],
    )


# =========================================================
# 7. Story 초기화
# =========================================================
def reset_story():
    """현재 이야기와 LangChain 대화 기록을 초기화합니다."""

    old_session_id = st.session_state["llm_session_id"]
    clear_session_history(old_session_id)

    st.session_state["data_dict"] = {}
    st.session_state["oid_list"] = []
    st.session_state["llm_session_id"] = str(uuid.uuid4())
    st.session_state["story_genre"] = ""
    st.session_state["story_started"] = False


# =========================================================
# 8. Story Part 출력
# =========================================================
def render_story_parts():
    """이미 생성된 모든 Story Part를 순서대로 화면에 출력합니다."""

    oid_list = list(st.session_state["oid_list"])

    for index, oid in enumerate(oid_list, start=1):
        data = st.session_state["data_dict"][oid]

        # 마지막 Part만 선택 가능
        is_last_part = oid == oid_list[-1]

        with st.expander(
            f"Part {index}",
            expanded=is_last_part,
        ):
            col1, col2 = st.columns([0.65, 0.35])

            with col1:
                st.write(data["story"])

                # 마지막 Part에서만 다음 선택을 받습니다.
                if is_last_part:
                    with st.form(key=f"choice_form_{oid}"):
                        selected_choice = st.radio(
                            data["decisionQuestion"],
                            data["choices"],
                            key=f"radio_{oid}",
                        )

                        next_clicked = st.form_submit_button(
                            "진행하기"
                        )

                    if next_clicked:
                        try:
                            create_next_part(selected_choice)

                        except Exception as exc:
                            st.error(
                                f"다음 이야기를 생성하지 못했습니다.\n\n{exc}"
                            )

                        else:
                            st.rerun()

                else:
                    st.caption("선택이 완료된 이전 이야기입니다.")

            with col2:
                if data["img"] is not None:
                    st.image(
                        data["img"],
                        use_container_width=True,
                    )


# =========================================================
# 9. Main
# =========================================================
def main():
    init_session_state()

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------
    with st.sidebar:
        st.header("📚 NovelGPT")

        st.markdown(
            """
NovelGPT는 사용자의 선택에 따라 이야기가 달라지는
인터랙티브 스토리 생성 애플리케이션입니다.

- **OpenAI GPT 모델**: 스토리 생성
- **GPT Image 모델**: 장면 이미지 생성
- **LangChain**: 대화 문맥 관리
"""
        )

        st.info("OpenAI API Key를 입력하세요.")

        # 입력 위젯의 값과 실제 API 호출용 값을 분리합니다.
        # api_key_input   : 화면 입력용
        # openai_api_key  : 실제 API 호출용 영구 저장값
        with st.form("api_key_form"):
            st.text_input(
                "OpenAI API Key",
                key="api_key_input",
                type="password",
                disabled=st.session_state["api_authenticated"],
            )

            api_submit = st.form_submit_button(
                "Submit",
                disabled=st.session_state["api_authenticated"],
            )

        if api_submit:
            api_key = st.session_state["api_key_input"].strip()

            if not api_key:
                st.error("OpenAI API Key를 입력하세요.")

            else:
                # 입력 위젯의 값을 별도의 session_state 변수에 복사하여
                # rerun 이후에도 실제 API 호출에서 사용할 수 있게 합니다.
                st.session_state["openai_api_key"] = api_key
                st.session_state["api_authenticated"] = True
                st.rerun()

        if st.session_state["api_authenticated"]:
            st.success("API Key가 입력되었습니다.")

        with st.expander("사용 가이드"):
            st.markdown(
                """
1. OpenAI API Key를 입력하고 **Submit**을 누릅니다.
2. 이야기의 주제 또는 주인공을 입력합니다.
3. **시작!** 버튼을 누릅니다.
4. 생성된 선택지 중 하나를 선택합니다.
5. **진행하기** 버튼으로 이야기를 이어갑니다.
"""
            )

    # -----------------------------------------------------
    # Main UI
    # -----------------------------------------------------
    st.title("📚 NovelGPT")

    if (
        not st.session_state["api_authenticated"]
        or not st.session_state["openai_api_key"].strip()
    ):
        st.warning(
            "OpenAI API Key가 입력되지 않았습니다.",
            icon="⚠️",
        )

    input_disabled = (
        not st.session_state["api_authenticated"]
        or not st.session_state["openai_api_key"].strip()
        or st.session_state["story_started"]
    )

    col_1, col_2, col_3 = st.columns(
        [8, 1, 1],
        gap="small",
    )

    col_1.text_input(
        "이야기의 주제 또는 주인공을 입력하세요.",
        key="genre_input",
        placeholder="예: 아기 펭귄 보물이의 모험",
        disabled=input_disabled,
    )

    start_clicked = col_2.button(
        "시작!",
        disabled=input_disabled,
        use_container_width=True,
    )

    reset_clicked = col_3.button(
        "새로 시작",
        disabled=not st.session_state["story_started"],
        use_container_width=True,
    )

    if reset_clicked:
        reset_story()
        st.rerun()

    # -----------------------------------------------------
    # 최초 Story 생성
    # -----------------------------------------------------
    if start_clicked:
        genre = st.session_state["genre_input"].strip()

        if not genre:
            st.error("이야기의 주제 또는 주인공을 입력하세요.")

        else:
            # 생성이 성공한 뒤에만 story_started=True로 변경합니다.
            st.session_state["story_genre"] = genre

            try:
                create_next_part(genre)

            except Exception as exc:
                # 실패하면 다시 시작할 수 있도록 원상복구합니다.
                st.session_state["story_genre"] = ""

                st.error(
                    f"첫 번째 이야기를 생성하지 못했습니다.\n\n{exc}"
                )

            else:
                st.session_state["story_started"] = True
                st.rerun()

    # -----------------------------------------------------
    # 생성된 Story Part 출력
    # -----------------------------------------------------
    if st.session_state["oid_list"]:
        render_story_parts()


if __name__ == "__main__":
    main()
