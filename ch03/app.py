import streamlit as st
import pandas as pd
import numpy as np

# 페이지 기본설정
st.set_page_config(page_title="일별 매출 대시보드", layout="wide")

# 1. 앱 제목
st.title("📊 일별 매출 및 방문자 대시보드")
st.caption("Streamlit으로 구현한 간단한 종합 대시보드 예제입니다.")

# 2. 사이드바 - 사용자 입력 제어
st.sidebar.header("🔍 데이터 필터 설정")
user_name = st.sidebar.text_input("담당자 이름", "김파이썬")
data_days = st.sidebar.slider("조회 일수 선택", min_value=7, max_value=60, value=30)

# 3. 가상 데이터 생성 (난수 고정)
np.random.seed(42)
dates = pd.date_range(end=pd.Timestamp.today(), periods=data_days)
sales = np.random.randint(100, 500, size=data_days) * 1000
visitors = np.random.randint(50, 300, size=data_days)

df = pd.DataFrame({
    "날짜": dates,
    "매출액(원)": sales,
    "방문자 수": visitors
})

# 4. 환영 문구 및 지표 요약 (st.metric 사용)
st.write(f"**{user_name}** 담당자님, 최근 {data_days}일간의 실적 요약입니다.")

m1, m2, m3 = st.columns(3)
m1.metric("총 매출액", f"{df['매출액(원)'].sum():,} 원")
m2.metric("일평균 매출액", f"{int(df['매출액(원)'].mean()):,} 원")
m3.metric("총 방문자 수", f"{df['방문자 수'].sum():,} 명")

st.divider()

# 5. 본문 레이아웃 - 데이터 표 및 시각화 차트
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📈 매출 추이 그래프")
    # 날짜를 인덱스로 설정하여 라인 차트 생성
    st.line_chart(df.set_index("날짜")["매출액(원)"])

with right_col:
    st.subheader("📋 상세 데이터 목록")
    st.dataframe(df, use_container_width=True)