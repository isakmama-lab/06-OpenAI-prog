import os
import urllib.request

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# 1. 환경 설정
# =========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. 프로젝트의 .env 파일을 확인하세요.")
    st.stop()


# =========================================================
# 2. 기본 설정
# =========================================================

PDF_URL = (
    "https://github.com/chatgpt-kr/openai-api-tutorial/raw/main/ch07/"
    "2020_%EA%B2%BD%EC%A0%9C%EA%B8%88%EC%9C%B5%EC%9A%A9%EC%96%B4%20700%EC%84%A0_"
    "%EA%B2%8C%EC%8B%9C.pdf"
)
PDF_PATH = "2020_경제금융용어 700선_게시.pdf"


# =========================================================
# 3. RAG 시스템 생성
# =========================================================

@st.cache_resource(show_spinner="금융 용어 PDF를 읽고 벡터 DB를 생성하고 있습니다...")
def build_rag_system():

    # PDF가 없을 때만 다운로드
    if not os.path.exists(PDF_PATH):
        urllib.request.urlretrieve(PDF_URL, filename=PDF_PATH)

    # PDF 로드
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    #  긴 문서를 검색 가능한 작은 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter()

    texts = text_splitter.split_documents(documents)

    # 앞부분 목차 제거
    texts = texts[13:]

    # 마지막 데이터 제거
    texts = texts[:-1]

    # OpenAI Embedding
    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key
    )

    # Chroma Vector DB
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding
    )

    # 유사도가 높은 문서 2개 검색
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 2}
    )

    # 금융 챗봇 Prompt 유지
    template = """당신은 한국은행에서 만든 금융 용어를 설명해주는 금융쟁이입니다.
안상준 개발자가 만들었습니다. 주어진 검색 결과를 바탕으로 답변하세요.
검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 반말로 친근하게 답변하세요.
{context}

Question: {question}
Answer:
"""

    prompt = PromptTemplate.from_template(template)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=api_key
    )

    # LangChain Runnable 파이프라인
    answer_chain = prompt | llm | StrOutputParser()

    return retriever, answer_chain


retriever, answer_chain = build_rag_system()


# =========================================================
# 4. RAG 답변 생성
# =========================================================

def get_chatbot_response(input_text):

    # 1) 질문과 유사한 문서 검색
    docs = retriever.invoke(input_text)

    # 2) 검색 문서를 Context로 결합
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # 3) 검색 결과를 근거로 LLM 답변 생성
    result = answer_chain.invoke(
        {
            "context": context,
            "question": input_text
        }
    )

    return result.strip()


# =========================================================
# 5. Streamlit UI
# =========================================================

st.set_page_config(
    page_title="경제금융용어 챗봇",
    page_icon="💰"
)

st.title("💰 경제금융용어 챗봇")
st.caption("한국은행 『경제금융용어 700선』을 기반으로 답변하는 RAG 챗봇입니다.")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if user_input := st.chat_input("금융 용어를 질문해주세요."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("관련 금융 용어를 검색하고 있습니다..."):
            bot_message = get_chatbot_response(user_input)

        st.markdown(bot_message)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": bot_message
        }
    )

# 대화 초기화
if st.button("대화 초기화"):
    st.session_state.messages = []
    st.rerun()

 