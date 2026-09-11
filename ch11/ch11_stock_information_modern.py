"""Modern Responses API implementation for the stock chatbot."""
import json, os
from pathlib import Path
import openai
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

@st.cache_resource
def get_client():
    key = os.getenv("OPENAI_API_KEY")
    try:
        key = key or st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    if not key:
        st.error("OPENAI_API_KEY를 환경변수 또는 .streamlit/secrets.toml에 설정하세요.")
        st.stop()
    return openai.OpenAI(api_key=key)

@st.cache_data(ttl=300)
def get_price(symbol):
    x=yf.Ticker(symbol.upper()).fast_info
    return {"symbol":symbol.upper(),"price":float(x["last_price"]),"currency":x.get("currency","USD")}

@st.cache_data(ttl=600)
def get_news(symbol):
    result=[]
    for x in yf.Ticker(symbol.upper()).news[:3]:
        c=x.get("content",x); u=c.get("canonicalUrl",{}).get("url",c.get("link",""))
        result.append({"title":c.get("title",""),"publisher":c.get("publisher",""),"link":u})
    return result

INSTRUCTIONS="항상 한국어로만 답변하세요. 도구 결과가 영어여도 주가와 뉴스 내용을 반드시 한국어로 설명하세요."
TOOLS=[{"type":"function","name":"get_price","description":"현재 주가를 조회합니다.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"],"additionalProperties":False}},{"type":"function","name":"get_news","description":"최신 기업 뉴스를 조회합니다.","parameters":{"type":"object","properties":{"symbol":{"type":"string"}},"required":["symbol"],"additionalProperties":False}}]

def ask(question):
    r=get_client().responses.create(model="gpt-4o-mini",instructions=INSTRUCTIONS,input=f"반드시 한국어로 답변하세요. 질문: {question}",tools=TOOLS)
    for _ in range(3):
        calls=[x for x in r.output if x.type=="function_call"]
        if not calls:return r.output_text
        outputs=[]
        for c in calls:
            a=json.loads(c.arguments); v=get_price(a["symbol"]) if c.name=="get_price" else get_news(a["symbol"])
            outputs.append({"type":"function_call_output","call_id":c.call_id,"output":json.dumps(v,ensure_ascii=False)})
        r=get_client().responses.create(
            model="gpt-4o-mini",
            instructions=INSTRUCTIONS,
            input=[*r.output, *outputs],
            tools=TOOLS,
        )
    return "응답을 완료하지 못했습니다."

def main():
    st.set_page_config(page_title="주식 정보 AI 챗봇")
    st.title("주식 정보 AI 챗봇")
    if "messages" not in st.session_state: st.session_state.messages=[]
    if st.sidebar.button("대화 초기화",width="stretch"): st.session_state.messages=[]; st.rerun()
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if q:=st.chat_input("예: AAPL 현재 주가와 최신 뉴스를 알려줘"):
        st.session_state.messages.append({"role":"user","content":q})
        with st.chat_message("user"): st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("조회 중..."): a=ask(q)
            st.markdown(a)
        st.session_state.messages.append({"role":"assistant","content":a})

if __name__=="__main__": main()
