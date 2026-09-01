import streamlit as st
import pandas as pd

# 1. 텍스트 요소
st.title("Streamlit 기초 배우기")
st.write("`st.write()`는 텍스트, 데이터프레임, 차트 등을 자동으로 알맞게 출력해 줍니다.")

# 2. 사용자 입력 위젯
name = st.text_input("이름을 입력하세요:", "홍길동")
age = st.slider("나이를 선택하세요:", 1, 100, 25)

if st.button("인사하기"):
    st.success(f"안녕하세요, {name}님! 나이는 {age}세이시군요.")

# 3. 데이터프레임 표시
data = {
    "과일": ["사과", "바나나", "포도"],
    "수량": [10, 15, 7]
}
df = pd.DataFrame(data)
st.dataframe(df)

# 사이드바 사용 예시
st.sidebar.header("설정 메뉴")
option = st.sidebar.selectbox("선호하는 음료는?", ["커피", "차", "주스"])

# 컬럼 레이아웃 사용 예시
col1, col2 = st.columns(2)

with col1:
    st.write("첫 번째 열입니다.")
    st.info(f"선택한 음료: {option}")

with col2:
    st.write("두 번째 열입니다.")
    st.warning("경고 메시지 예시입니다.")