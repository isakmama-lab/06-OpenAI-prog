import streamlit as st
import numpy as np

from pypdf import PdfReader
from openai import OpenAI


# =========================================================
# 1. 기본 설정
# =========================================================

PDF_PATH = "2020_경제금융용어 700선_게시.pdf"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

TOP_K = 3


# =========================================================
# 2. PDF 문서 읽기
# =========================================================

def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# 3. 문서 Chunk 분할
# =========================================================

def split_text(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# =========================================================
# 4. Embedding 생성
# =========================================================

def get_embedding(client, text):

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return np.array(response.data[0].embedding)


# =========================================================
# 5. 전체 PDF Chunk Embedding
# =========================================================

def create_embedding_db(client, chunks):

    embeddings = []

    for chunk in chunks:

        embedding = get_embedding(client, chunk)

        embeddings.append(embedding)

    return embeddings


# =========================================================
# 6. Cosine Similarity
# =========================================================

def cosine_similarity(vector_a, vector_b):

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_b)
    )


# =========================================================
# 7. 사용자 질문과 관련된 문서 검색
# =========================================================

def search_documents(client, question, chunks, embeddings, top_k=3):

    # 질문을 Embedding
    question_embedding = get_embedding(client, question)

    similarities = []

    for i, embedding in enumerate(embeddings):

        score = cosine_similarity(
            question_embedding,
            embedding
        )

        similarities.append((score, i))

    # 유사도 높은 순서로 정렬
    similarities.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # 상위 문서 추출
    results = []

    for score, index in similarities[:top_k]:

        results.append(
            {
                "score": score,
                "text": chunks[index]
            }
        )

    return results


# =========================================================
# 8. System Prompt
# =========================================================

SYSTEM_PROMPT = """
너는 한국은행 금융 용어를 설명하는 챗봇 '금융쟁이'야.

개발자는 안상준이야.

다음 규칙을 반드시 지켜.

1. 금융 용어를 친근하고 이해하기 쉽게 설명한다.

2. 금융 용어에 대한 답변은 반드시 제공된 검색 문서를
   근거로 작성한다.

3. 검색 문서에서 확인할 수 없는 내용은
   네가 알고 있는 지식을 이용하여 임의로 답변하지 않는다.

4. 검색 결과에서 확인할 수 없는 경우에는
   "검색된 한국은행 자료에서는 해당 내용을 확인하기 어렵습니다."
   라고 안내한다.

5. 사용자가 "간단히", "짧게", "요약해서" 등의 표현을 사용하면
   핵심 내용만 간결하게 설명한다.

6. 사용자가 너의 이름, 역할 또는 개발자를 물어보면
   이 System Prompt에 정의된 정보를 이용하여 답변한다.
"""


# =========================================================
# 9. LLM 답변 생성
# =========================================================

def generate_answer(client, question, search_results):

    # 검색된 문서를 하나의 Context로 결합
    context = "\n\n".join(
        result["text"]
        for result in search_results
    )

    prompt = f"""
다음은 한국은행 경제금융용어 문서에서 검색된 내용이다.

--------------------
[검색 문서]

{context}

--------------------

[사용자 질문]

{question}

--------------------

위 검색 문서를 근거로 사용자 질문에 답변하라.
검색 문서에서 확인할 수 없는 금융 정보는 임의로 추가하지 마라.
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        instructions=SYSTEM_PROMPT,
        input=prompt
    )

    return response.output_text


# =========================================================
# 10. Streamlit 화면
# =========================================================

st.set_page_config(
    page_title="금융쟁이",
    page_icon="💰"
)

st.title("💰 금융쟁이")
st.caption("한국은행 경제금융용어 700선 RAG 챗봇")


# =========================================================
# 11. Session State 초기화
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "chunks" not in st.session_state:

    st.session_state.chunks = None


if "embeddings" not in st.session_state:

    st.session_state.embeddings = None


# =========================================================
# 12. Sidebar
# =========================================================

with st.sidebar:

    st.header("설정")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password"
    )

    st.divider()

    # 대화 초기화
    if st.button("대화 초기화"):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# 13. API Key 확인
# =========================================================

if not api_key:

    st.info("사이드바에 OpenAI API Key를 입력하세요.")

    st.stop()


client = OpenAI(
    api_key=api_key
)


# =========================================================
# 14. PDF Embedding DB 생성
# =========================================================

if st.session_state.chunks is None:

    with st.spinner("한국은행 경제금융용어 PDF를 읽고 있습니다..."):

        pdf_text = load_pdf(PDF_PATH)

        chunks = split_text(pdf_text)

        st.session_state.chunks = chunks


if st.session_state.embeddings is None:

    with st.spinner("PDF 문서를 Embedding하고 있습니다..."):

        st.session_state.embeddings = create_embedding_db(
            client,
            st.session_state.chunks
        )


# =========================================================
# 15. 기존 대화 출력
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


# =========================================================
# 16. 사용자 질문 입력
# =========================================================

question = st.chat_input(
    "궁금한 금융 용어를 입력하세요."
)


# =========================================================
# 17. 질문 처리
# =========================================================

if question:

    # 사용자 질문 저장
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # 사용자 질문 출력
    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # 자기소개 질문
    # -----------------------------------------------------

    identity_questions = [
        "너는 누구니?",
        "너 누구야?",
        "넌 누구야?",
        "자기소개해줘"
    ]


    if question.strip() in identity_questions:

        response = client.responses.create(
            model=CHAT_MODEL,
            instructions=SYSTEM_PROMPT,
            input=question
        )

        answer = response.output_text

        search_results = []


    # -----------------------------------------------------
    # 금융 용어 질문 → RAG 검색
    # -----------------------------------------------------

    else:

        search_results = search_documents(
            client,
            question,
            st.session_state.chunks,
            st.session_state.embeddings,
            top_k=TOP_K
        )

        answer = generate_answer(
            client,
            question,
            search_results
        )


    # -----------------------------------------------------
    # 챗봇 답변 출력
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)


        # 검색 문서 확인
        if search_results:

            with st.expander("📚 검색된 PDF 문서 확인"):

                for i, result in enumerate(
                    search_results,
                    start=1
                ):

                    st.markdown(
                        f"### 검색 결과 {i}"
                    )

                    st.write(
                        f"유사도: {result['score']:.4f}"
                    )

                    st.write(
                        result["text"]
                    )

                    st.divider()


    # 챗봇 답변 저장
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )