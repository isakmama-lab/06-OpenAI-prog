import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "2020_경제금융용어 700선.pdf"

# 실행 위치와 관계없이 Python 파일이 있는 폴더의 상위 폴더에서 .env를 읽는다.
ENV_PATH = BASE_DIR.parent / ".env"

load_dotenv(dotenv_path=ENV_PATH)


def check_api_key():
    """.env에서 OpenAI API Key가 로드되었는지 확인한다."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            f"{ENV_PATH} 파일에 OPENAI_API_KEY를 설정하세요."
        )


def prepare_documents():
    """로컬 PDF를 읽고 금융용어 본문 페이지만 반환한다."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"PDF 파일을 찾을 수 없습니다: {PDF_PATH}"
        )

    loader = PyPDFLoader(str(PDF_PATH))
    texts = loader.load_and_split()

    # 원본 수업 코드 기준: 0~12번은 머리말·목차, 마지막은 불필요한 페이지
    return texts[13:-1]


def create_qa_chain(texts):
    """Embedding, Chroma, Retriever, Prompt, LLM을 RAG 체인으로 연결한다."""
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=texts, embedding=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    template = """당신은 한국은행에서 만든 금융 용어를 설명해주는 금융쟁이입니다.
안상준 개발자가 만들었습니다. 주어진 검색 결과를 바탕으로 답변하세요.
검색 결과에 없는 내용이라면 답변할 수 없다고 하세요. 반말로 친근하게 답변하세요.
{context}

Question: {question}
Answer:
"""
    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


@st.cache_resource(show_spinner=False)
def load_qa_chain():
    """Streamlit이 재실행되어도 임베딩과 Vector DB를 다시 만들지 않는다."""
    check_api_key()
    documents = prepare_documents()
    return create_qa_chain(documents)


def get_chatbot_response(qa_chain, input_text):
    """질문을 RAG 체인에 전달하고 답변 문자열을 반환한다."""
    chatbot_response = qa_chain.invoke(input_text)
    return chatbot_response["result"].strip()


def run_streamlit_app(qa_chain):
    """Streamlit 채팅 화면과 대화 상태를 구성한다."""
    st.title("💰 경제금융용어 챗봇")
    st.caption("한국은행 『경제금융용어 700선』 기반 RAG 챗봇")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("챗봇 설정")
        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    input_text = st.chat_input("금융 용어를 질문해주세요!")

    if input_text:
        st.session_state.messages.append(
            {"role": "user", "content": input_text}
        )
        with st.chat_message("user"):
            st.markdown(input_text)

        with st.chat_message("assistant"):
            with st.spinner("관련 문서를 검색하고 있어요..."):
                answer = get_chatbot_response(qa_chain, input_text)
            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )


if __name__ == "__main__":
    st.set_page_config(page_title="경제금융용어 챗봇", page_icon="💰")

    try:
        chain = load_qa_chain()
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
        st.stop()

    run_streamlit_app(chain)
