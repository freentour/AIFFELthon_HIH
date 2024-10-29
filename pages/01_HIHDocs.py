from typing import List, Tuple, Dict, Any
import re

from dotenv import load_dotenv
from langchain_teddynote import logging

from langchain_community.document_loaders import TextLoader

# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import ChatUpstage, UpstageEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_core.callbacks import BaseCallbackHandler

import streamlit as st
from datetime import datetime

# -----

from code317.code317kiwi import Code317Kiwi
from code317.code317rag import Code317Rag, code_pretty_print

# -----

# from langchain_openai import ChatOpenAI
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough


# Streaming 처리를 위한 콜백 핸들러(streamlit 전용)
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# LLM 객체 생성을 위한 공통 함수
def get_llm(code_rag: Code317Kiwi, streaming: bool = False):
    # chat_llm 객체 생성
    if code_rag.vendor == "Google":
        llm = ChatGoogleGenerativeAI(
            model=code_rag.model,
            temperature=code_rag.temperature,
            # streaming=streaming,      # Google 모델의 경우 streaming 지원 방식 체크 필요!
        )
    elif code_rag.vendor == "Anthropic":
        if streaming:
            llm = ChatAnthropic(
                model=code_rag.model,
                temperature=code_rag.temperature,
                streaming=True,
                callbacks=[
                    ChatCallbackHandler(),
                ],
            )
        else:
            llm = ChatAnthropic(
                model=code_rag.model,
                temperature=code_rag.temperature,
            )
    elif code_rag.vendor == "upstage":
        llm = ChatUpstage(
            model=code_rag.model,
            temperature=code_rag.temperature,
            # streaming=streaming,      # upstage 모델의 경우 streaming 지원 방식 체크 필요!
        )
    else:   # 기본 모델은 OpenAI 모델
        if streaming:
            llm = ChatOpenAI(
                model=code_rag.model,
                temperature=code_rag.temperature,
                streaming=True,
                callbacks=[
                    ChatCallbackHandler(),
                ],
            )
        else:
            llm = ChatOpenAI(
                model=code_rag.model,
                temperature=code_rag.temperature,
            )
    
    return llm


# >>>>> [시작] RAG 사전 준비 영역(캐싱 대상)

# [주의] 파라미터 없는 함수에서 st.cache_data 데코레이터가 의미 있으려면 return하는 결과가 항상 동일해야 함!
# 왜냐하면, 한 번 캐시되면 return 결과가 달라져도 무조건 사전에 캐싱된 결과가 return되기 때문임. 
# 만약, return 결과가 달라지는 상황이면 무조건 return 결과가 달라지는데 영향을 주는 요인들을 파라미터로 선언해서 사용해야 함. 
# 파라미터가 있는 함수에서는 파라미터의 값이 변경되지 않으면 기존에 캐싱해둔 결과값을 return하게 됨. 
# [참고] @st.cache_data 데코레이터 사용하면 pickle 형식으로 serialize할 수 없다고 에러 남! 그래서, 대신 cache_resource 사용함!!
# @st.cache_resource(ttl="5m", show_spinner="RAG 서비스 준비중..")      # 5분동안만 캐싱하고 싶을 때 ttl 사용

@st.cache_resource()
def set_env():
    # ----- 0. 환경 설정 -----
    load_dotenv()   # API KEY 정보 로드
    logging.langsmith("HIH-RAG-v0.5")   # langsmith 프로젝트 이름


@st.cache_resource(show_spinner="RAG 서비스 준비중..")
def prepare_rag():
    st.sidebar.write("RAG 서비스 준비 시작..")
    today = datetime.today().strftime("%H:%M:%S")
    st.sidebar.write(today)

    # 한입해 RAG 전용으로 정의한 Code317Rag 객체 생성
    code_rag = Code317Rag()     # RAG 초기화 작업만 진행: 프로퍼티 초기화, load_dotenv()만 실행.
    vectorstore = code_rag.prepare_rag()  # RAG 서비스에 필요한 사전 작업 완료 후 최종적으로 vectorstore 리턴

    # RAG에서 사용할 기본 LLM 모델과 temperature 세팅 
    # -> 이건 static하게 설정하고 캐싱하기 보다는 execute_conditional_rag 함수 호출할 때 설정하는 것이 더 유연하기 때문에 여기서는 주석처리함.
    # code_rag.set_model_hyperparams(model_str="GPT-4o", temperature=0.1)

    # ChatPromptTemplate 객체 생성
    chat_prompt_message = [
        (
            "system", 
            """
            너는 한국의 대학 입시와 진학 지도에 대한 전문가야. 오직 주어진 문서 내용만 참고해서 질문에 해당하는 최종 답변을 작성해 줘. 답변은 항상 존대말로 해 줘. 

            [규칙]
            - 문서의 내용 중에서 답변과 관련 있는 부분은 절대로 요약하거나 생략하지 마. 
            - 절대 주어진 문서에 없는 내용으로 답을 지어내려고 노력하지 마. 주어진 문서 자체가 없거나 주어진 문서에서 답을 찾을 수 없으면 그냥 찾을 수 없다고 말해. 
            - 목록 형식으로 정리해서 답변할 수 있는 내용은 가급적 문장 형식보다는 목록 형식으로 답변해 줘. 
            - 너가 답변을 생성하는데 있어서 유효하게 참고한 문서가 있는 경우에만 답변의 마지막에 blank line을 하나 추가한 다음 참고한 문서의 title과 page 번호를 참고해 '[출처] ' 문자열과 concate해서 함께 답변해 줘. page 번호는 너가 참고한 문서 내의 <page> element 내용을 참고하고, title 정보는 Metadata의 title 정보를 참고해. (예시: title 정보가 '대입뉴스'이고, page 번호가 5와 7인 경우라면, '[출처] 대입뉴스 | 5,7페이지')
            - 답변 생성에 사용된 문서만 출처에 포함시키고, 사용되지 않은 문서는 절대로 출처에 포함시키면 안되. 
            - 주어진 문서 자체가 없거나 주어진 문서에서 답을 찾을 수 없거나 답변할 내용이 거의 없는 경우에는 출처 정보를 무조건 생략하고 답변해. 
            - '지원전략'에 대한 내용을 답변에 포함할 때는 문서에 포함된 내용을 생략하거나 요약, 변경하지 말고 원본 문서에 포함된 '지원전략' 내용을 그대로 답변해 줘. 
            - 문서에 포함된 내용이 아닌 너의 opinion(의견)에 대해 질문받는 경우에는, 먼저 문서에 포함된 내용을 답변하고 마지막에 너의 opinion은 '(의견) '이라는 머리글로 시작해서 답변해 줘. 
            -----
            {context}
            """
        ),
        (
            "human", 
            "{question}"
        )
    ]
    chat_prompt = code_rag.get_chat_prompt_template(chat_prompt_message)

    # st.sidebar.write("RAG 서비스 준비 종료..")    # 제일 마지막 스텝 완료 후 출력
    return code_rag, vectorstore, chat_prompt


# 사용자 질문의 형태소 분석을 위해 사용되는 Code317Kiwi 객체 가져오기(사용자 질문 의도 분석용)
# [중요] 사용자의 모든 질문에 대해 사용되기 때문에 반드시 캐싱해 두어야 함. 그래야 형태소 분석 시간이 짧아짐. 
@st.cache_resource
def get_codekiwi() -> Code317Kiwi:
    # 형태소 분석에 사용되는 Kiwi 클래스를 상속받아 커스터마이징한 클래스
    # [중요] Kiwi 객체는 모든 사용자의 질문에 대해서 사용되기 때문에 캐싱해 두어야 함. 그래야 형태소 분석 시간이 짧아짐.
    code_kiwi = Code317Kiwi()
    return code_kiwi


# DB 기반 RAG와 WEB 기반 RAG에서 사용할 데이터베이스 객체 가져오기
# [중요] 데이터베이스 연결 역시 매우 자주 사용되기 때문에 반드시 캐싱해 두어야 함. 
@st.cache_resource
def get_databases():
    # DB 기반 RAG 전용 SQLite 데이터베이스 연결
    db = SQLDatabase.from_uri("sqlite:///databases/haniphae.db", max_string_length=0)

    # WEB 기반 RAG 전용 SQLite 데이터베이스 연결
    # [중요] max_string_length 옵션을 0 또는 음수로 주어 contents 컬럼과 같이 텍스트 길이가 매우 긴 경우 잘림 현상이 발생하지 않도록 해야 함!
    # 지정하지 않으면 기본값 300이 적용되어 300글자 이후로는 잘리고 '...'이 마지막에 추가되도록 내부에서 자동으로 처리함. (즉, 전체 데이터 확인이 안됨)
    news_db = SQLDatabase.from_uri("sqlite:///databases/haniphae_news.db", max_string_length=0)

    return db, news_db


# DB 기반 RAG를 위한 사전 작업
@st.cache_resource
def prepare_db_rag():
    # 한입해 RAG 전용으로 정의한 Code317Rag 객체 생성
    code_rag = Code317Rag()     # RAG 초기화 작업만 진행: 프로퍼티 초기화, load_dotenv()만 실행.
    # RAG에서 사용할 기본 LLM 모델과 temperature 세팅 
    # -> 이건 static하게 설정하고 캐싱하기 보다는 execute_conditional_rag 함수 호출할 때 설정하는 것이 더 유연하기 때문에 여기서는 주석처리함.
    # code_rag.set_model_hyperparams(model_str="GPT-4o mini", temperature=0)

    # PromptTemplate 객체 생성
    sql_prompt_message = """
    You are a SQLite expert. Given an input question, first create ONLY the SQL query without any prefixes, explanations, additional text, or code formatting symbols to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, do not limit the number of results {top_k} returned by the query. You can order the results to return the most informative data in the database.
    When creating the SQL query, Always include all columns except column 'id' from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use date('now') function to get the current date, if the question involves "today". When creating a query statement, be sure to include '지역', '대학명', '연도', '전형유형', '전형명', '계열', '학부' and '모집단위'.
    When using where clauses, use % instead of =. When searching for a department or school, use the or clause to include both '학부' and '모집단위'.

    Use the following format:

    Question: Question here
    SQLResult: Result of the SQLQuery
    Answer: Final answer here

    Only use the following tables:
    {table_info}

    Question: {input}
    """
    sql_prompt = code_rag.get_prompt_template(sql_prompt_message)

    # ChatPromptTemplate 객체 생성
    answer_prompt_message = """
    SQL 결과가 주어지면 사용자 질문에 답해주는데 모든 답변은 한글로 해주고 DB에 데이터가 없다면 없다고 정중하게 대답해줘.
    동일한 연도에 여러 값이 있으면 다 작성해서 답변해줘.
    통합선발에 대한 내용은 괄호 안에 있는 데이터(ex 국어국문학과, 사학과 등 )도 함께 대답해줘
    비교하는 질문을 받았을 때 둘 중 없는 데이터는 없다고 답변해주고 있는 데이터는 설명해줘.
    DB결과가 없으면 그 부분은 제외하고 답변해줘.
    질문이 명확하지 않으면 다시 질문해달라고 답변해줘.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer:
    """
    answer_prompt = code_rag.get_prompt_template(answer_prompt_message)

    return code_rag, sql_prompt, answer_prompt


# WEB 기반 RAG를 위한 사전 작업
@st.cache_resource
def prepare_web_rag():
    # 한입해 RAG 전용으로 정의한 Code317Rag 객체 생성
    code_rag = Code317Rag()     # RAG 초기화 작업만 진행: 프로퍼티 초기화, load_dotenv()만 실행.
    # RAG에서 사용할 기본 LLM 모델과 temperature 세팅 
    # -> 이건 static하게 설정하고 캐싱하기 보다는 execute_conditional_rag 함수 호출할 때 설정하는 것이 더 유연하기 때문에 여기서는 주석처리함.
    # code_rag.set_model_hyperparams(model_str="GPT-4o mini", temperature=0)

    # PromptTemplate 객체 생성
    sql_prompt_message = """
    You are a SQLite expert. Your task is to generate a syntactically correct SQLite query based on the given input question.

    Follow these rules:
    1. Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per SQLite. 
    2. You can order the results to return the most informative data in the database.
    3. When creating the SQL query, Always include all columns except column 'id' from a table.
    4. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    5. Pay attention to use only the column names you can see in the tables below.
    6. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    7. Pay attention to use date('now') function to get the current date, if the question involves "today". 
    8. 총 두 개의 query문을 작성한 뒤, 두 개의 query문을 UNION으로 연결해서 최종 query문을 만들어줘. 첫번째 query문의 WHERE 절을 구성할 때는 'title' 컬럼과만 비교하고, 두번째 query문의 WHERE 절을 구성할 때는 'contents' 컬럼과만 비교해줘. 
    9. LIMIT 절은 각 query마다 추가하지 말고, 전체 query문의 제일 마지막에 한 번만 붙여줘. 

    Provide ONLY the SQL query without any prefixes, explanations, additional text, or code formatting symbols.

    Only use the following tables:
    {table_info}

    Question: {input}
    """
    sql_prompt = code_rag.get_prompt_template(sql_prompt_message)

    # ChatPromptTemplate 객체 생성
    answer_prompt_message = """
    SQL 결과가 주어지면 해당 내용을 살펴보고 사용자 질문에 답해줘. 모든 답변은 한글로 해주고, SQL 결과가 없거나 SQL 결과는 있지만 해당 내용에서 사용자 질문에 대한 답변을 찾을 수 없는 경우에는 데이터를 찾을 수 없다고 정중하게 대답해줘. 절대 SQL 결과에 없는 내용을 답변하려고 해서는 안되.

    Follow these rules:
    1. 목록 형식으로 정리해서 답변할 수 있는 내용은 가급적 문장 형식보다는 목록 형식으로 답변해 줘.
    2. 답변할 내용이 여러 개의 기사를 목록 형태로 표시해야 하는 경우에는, 각 기사별로 'title', 'subheading', 'contents', 'date_input', 'date_update', 'link' 컬럼만 포함해 주고, 'contents' 컬럼의 내용은 특별한 지시가 있지 않으면 내용을 요약하거나 question과 연관있는 부분만 추출해서 답변해 줘. 'subheading', 'contents', 'date_input', 'date_update', 'link' 컬럼은 'title' 컬럼의 하위 목록으로 구성해서 답변해줘. 'title', 'subheading', 'contents', 'date_input', 'date_update', 'link' 컬럼 이름도 한글로 바꿔서 답변해줘. 
    3. 답변 마지막에 답변 내용을 구성하는데 있어서 실질적으로 참고한 뉴스의 'title'과 'link'를 '[출처]' 문자열 다음 줄에 목록 형태로 표시해줘. 답변할 내용이 여러 개의 기사를 목록 형태로 표시해야 하는 경우에는, 답변 마지막에 출처 정보 포함시키지 마.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer:
    """
    answer_prompt = code_rag.get_prompt_template(answer_prompt_message)

    st.sidebar.write("RAG 서비스 준비 종료..")    # 제일 마지막 스텝 완료 후 출력
    return code_rag, sql_prompt, answer_prompt


# >>>>> [끝] RAG 사전 준비 영역(캐싱 대상)


# >>>>> [시작] LCEL 및 메인 프로그램 영역

# 10. LCEL 이용해 체인 구성

# 사용자 질문에 따라 RAG 라우팅 처리하고 랭체인 실행하는 함수
def execute_conditional_rag(vectorstore: FAISS, query: str, model_str: str = "GPT-4o mini", temperature: float = 0.1) -> Tuple[bool, str]:
    # query 형태소 분석 후 명사와 외국어만 추출(중복된 단어는 제거)
    noun_list_no_duplicates = code_kiwi.extract_nouns_foreigner_from_query(query)
    # print("----- 명사, 외국어 추출 -----")
    # print(noun_list_no_duplicates)

    use = "문서"    # RAG 유형 변수 (기본값: '문서')

    # 추출한 단어 리스트를 활용해 RAG 유형 분류
    for noun in noun_list_no_duplicates:
        competition_rate_noun_list = ['경쟁률']
        if noun in competition_rate_noun_list:
            use = "DB"  # DB 기반 RAG로 라우팅 ON
            break
            # continue
        
        admission_results_noun_list = ['입시 결과', '입결', '합격자 성적', '50% 컷', '50% cut', '70% 컷', '70% cut', '충원 인원', '충원 비율', '충원율', '수능 최저 충족률', '최저 충족률', '충족률', '수능 최저 충족 인원', '최저 충족 인원', '충족 인원', '실질 경쟁률']
        if noun in admission_results_noun_list:
            use = "DB"  # DB 기반 RAG로 라우팅 ON
            break
            # continue
        
        # 전년도와 달라진 점, 경쟁률과 입시 결과 등 다른 섹션에서도 등장할 수 있으므로 필터링 우선순위 고려해야 함!
        # 전년도와 달라진 점, 경쟁률과 입시 결과 섹션이 우선 순위 더 높게 처리해야 함!!
        admission_unit_noun_list = ['모집 단위', '모집 인원']
        if noun in admission_unit_noun_list:
            use = "DB"  # DB 기반 RAG로 라우팅 ON
            break
            # continue

        news_noun_list = ['뉴스', '기사']
        if noun in news_noun_list:
            use = "WEB"  # WEB 기반 RAG로 라우팅 ON
            break
            # continue

    # '건국대 경쟁률 뉴스에서 검색해줘'의 경우 '경쟁률' 단어가 먼저 나오다보니 해당 규칙만 먼저 적용되어서 DB 기반 RAG로 플래그가 세팅되어진 채 for loop에서 break하게 됨.
    # 따라서, query에 '뉴스'나 '기사'라는 단어가 포함된 경우에는 강제로 WEB 기반 RAG로 재설정해 주는 과정이 필요함. 
    if any(word in query for word in ['뉴스', '기사']) and use != "WEB":
        use = "WEB"

    print("\n----- RAG 유형 분류 결과 -----")
    print(f"- RAG 유형: {use}")

    # 줄임말 풀네임으로 변환
    # 세번재 파라미터인 use(RAG 유형 변수) 값에 따라 내부에서 사용하는 변환 사전이 달라짐
    query, noun_list_no_duplicates = code_kiwi.replace_abbr_with_fullname(query, noun_list_no_duplicates, use)
    print(f"\n----- 줄임말 풀네임으로 변경({use} 기반) -----")
    print(f"- 변형된 사용자 query: {query}")
    print(f"- 풀네임으로 변경된 단어 리스트: {noun_list_no_duplicates}")

    if use == "문서":   # 문서 기반 RAG 경우인 경우에만 필터 목록 생성
        filter_dict = code_kiwi.construct_filter(noun_list_no_duplicates)   # 필터 목록 생성
        print(f"\n----- 필터 생성 결과 -----")
        print(f"- 필터 목록: {filter_dict}")

    direct_response_flag = False    # 랭체인 거치지 않고 곧바로 답변할지를 결정하는 용도

    # RAG 유형별로 라우팅 처리
    if use == "DB":
        # 사전에 진행되어야할 부분들은 모두 캐싱으로 처리하고 있음. 
        # 캐싱 단계에서의 처리 결과로 sql_db_prompt, answer_db_prompt가 이미 생성되어 있는 상황임. 
        
        # RAG에서 사용할 기본 LLM 모델과 temperature 세팅
        code_db_rag.set_model_hyperparams(model_str="GPT-4o mini", temperature=0)
        # code_db_rag.set_model_hyperparams(model_str="GPT-4o", temperature=0)
        # code_db_rag.set_model_hyperparams(model_str="Claude 3.5 Sonnet", temperature=0)

        # [주의] llm 객체는 캐싱해두면 절대 안됨! (특히, streaming 옵션이 켜져 있는 경우 미리 답변했던 내용이 계속 누적되어서 함께 나타남)
        # SQL 쿼리 생성용 LLM 모델 객체 생성
        sql_db_llm = get_llm(code_db_rag, streaming=False)    # SQL 생성용이라 streaming 하지 않음
        # 최종 답변용 LLM 모델 객체 생성
        answer_db_llm = get_llm(code_db_rag, streaming=True)    # 최종 답변용이라 streaming 옵션 ON

        write_query = create_sql_query_chain(sql_db_llm, db, prompt=sql_db_prompt)
        execute_query = QuerySQLDataBaseTool(db=db)
        answer_chain = answer_db_prompt | answer_db_llm | StrOutputParser()

        # 생성한 쿼리를 실행하고 결과를 출력하기 위한 체인을 생성합니다.
        chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer_chain
        )

        # RAG 체인 실행
        answer = chain.invoke({"question": query})
        # # model, temperature 정보와 함께 출력
        # answer = f"- model: {model}\n- temperature: {temperature}\n\n" + answer

        return direct_response_flag, answer
    elif use == "WEB":
        # 사전에 진행되어야할 부분들은 모두 캐싱으로 처리하고 있음. 
        # 캐싱 단계에서의 처리 결과로 sql_db_prompt, answer_db_prompt가 이미 생성되어 있는 상황임. 
        
        # RAG에서 사용할 기본 LLM 모델과 temperature 세팅
        # code_web_rag.set_model_hyperparams(model_str="GPT-4o mini", temperature=0)
        # code_web_rag.set_model_hyperparams(model_str="GPT-4o", temperature=0)
        code_web_rag.set_model_hyperparams(model_str="Claude 3.5 Sonnet", temperature=0)

        # [주의] llm 객체는 캐싱해두면 절대 안됨! (특히, streaming 옵션이 켜져 있는 경우 미리 답변했던 내용이 계속 누적되어서 함께 나타남)
        # SQL 쿼리 생성용 LLM 모델 객체 생성
        sql_web_llm = get_llm(code_web_rag, streaming=False)    # SQL 생성용이라 streaming 하지 않음
        # 최종 답변용 LLM 모델 객체 생성
        answer_web_llm = get_llm(code_web_rag, streaming=True)    # 최종 답변용이라 streaming 옵션 ON

        write_query = create_sql_query_chain(sql_web_llm, news_db, prompt=sql_web_prompt, k=5)
        execute_query = QuerySQLDataBaseTool(db=news_db)
        answer_chain = answer_web_prompt | answer_web_llm | StrOutputParser()

        # 생성한 쿼리를 실행하고 결과를 출력하기 위한 체인을 생성합니다.
        chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer_chain
        )

        # RAG 체인 실행
        answer = chain.invoke({"question": query})
        # # model, temperature 정보와 함께 출력
        # answer = f"- model: {model}\n- temperature: {temperature}\n\n" + answer

        return direct_response_flag, answer
    elif use == "문서":
        if not filter_dict:     # 필터 목록이 비어있으면
            if '이름' in query:
                direct_response_flag = True
                return direct_response_flag, "제 이름은 '꿀꺽'이예요. 모든 복잡한 입시 정보를 '한입에 꿀꺽'한다고 해서 꿀꺽~😋"
            elif '한입해' in query:
                direct_response_flag = True
                return direct_response_flag, "📢'한입해' 서비스는 입시 용어와 입시 정보가 낯선 분들을 위한 수시 정보 전문 챗봇으로, 24시간 아무 때나 입시에 관해 마음껏 질문하고 답변을 얻을 수 있는 서비스입니다. 궁금한 것이 있으면 언제든 질문해 주세요~🤗"
            else:
                direct_response_flag = True
                return direct_response_flag, "😅💦죄송합니다. 질문에 대한 답변 내용을 현재 제가 참고하고 있는 문서에서는 찾을 수 없습니다. 정상적인 질문인데도 제가 답변을 못하는 상황이라면 서비스 관리자 이메일로 문의 남겨주시면 빠르게 개선되도록 하겠습니다.💪"
        else:   # 필터 목록이 하나라도 채워져 있으면
            # [참고] 커스텀 클래스로 정의한 Code317FaissRetriever 역시 vectorsotre를 입력으로 받아서 similarity_search 메소드를 실행함. 
            # 차이는 기본적으로 similarity_search 메소드의 인자로 filter를 기본적으로 사용한다는 점 외에는 없음. 
            # 따라서, 별도로 커스텀 클래스를 정의해서 사용하기 보다는 그냥 similarity_search 메소드와 filter 인자를 직접 사용하면 됨. 
            # hih_faiss_retriever = Code317FaissRetriever(vectorstore=vectorstore, filter=filter_dict, k=6)

            # chunk 문서를 대상으로 맞춤형 필터링을 수행하고, metadata 중 필요한 부분만 포함되도록 조정하는 함수
            # [참고] LCEL의 RunnableLambda 함수에서 사용하는 함수는 입력 파라미터가 오직 하나만 가능!
            def retrieve_and_format_docs(query):
                # docs = hih_faiss_retriever.invoke(query)      # 커스텀 클래스를 사용하는 경우
                docs = vectorstore.similarity_search(query, filter=filter_dict, k=6)
                print("\n----- 필터링된 문서 목록 -----")
                print(f"- 문서 갯수: {len(docs)}")
                code_pretty_print(docs)

                formatted_docs = []
                pretty_print_list = []  # 출력용 문자열(문서 본문 중 처음 5줄만 출력)
                if not docs:  # 필터링된 문서 목록이 비어있는 경우
                    formatted_doc = '답변에 참고할 문서 없음.'
                    formatted_docs.append(formatted_doc)
                else:
                    for doc in docs:
                        # metadat 중 'source', 'title' 두 가지 항목만 추가
                        metadata_str = " | ".join([f"{k}: {v}" for k, v in doc.metadata.items() if k in ['source', 'title']])
                        formatted_doc = f"Content: \n{doc.page_content}\n\nMetadata: \n{metadata_str}\n"
                        formatted_docs.append(formatted_doc)
                        pretty_print_str = "\n".join(doc.page_content.splitlines()[:5]) + "\n... (나머지 생략) ..."
                        pretty_print_doc = f"Content: \n{pretty_print_str}\n\nMetadata: \n{metadata_str}\n"
                        pretty_print_list.append(pretty_print_doc)
                
                print("----- 최종적으로 LLM에게 전달되는 문서 목록 -----")
                print(f"- 문서 갯수: {len(formatted_docs)}")
                for doc in pretty_print_list:
                    print(doc)
                
                return "\n\n".join(formatted_docs)
            
            # RAG에서 사용할 LLM 모델, temperature 설정
            code_rag.set_model_hyperparams(model_str, temperature)
            # 채팅용 LLM 모델 객체 생성
            chat_llm = get_llm(code_rag, streaming=True)    # streaming 옵션 ON

            # 문서 기반 RAG: LCEL 사용해 chain 객체 생성
            chain = (
                {
                    # "context": retriever, 
                    # "context": ensemble_retriever,  # Ensemble Retriever로 변경
                    "context": RunnableLambda(lambda _: retrieve_and_format_docs(query)),   # 필터링된 문서 목록
                    # "question": RunnablePassthrough(),
                    "question": RunnableLambda(lambda _: query),
                } 
                | chat_prompt 
                | chat_llm
                | StrOutputParser()
            )

            # RAG 체인 실행
            answer = chain.invoke(query)
            # model, temperature 정보와 함께 출력
            answer = f"- model: {model_str}\n- temperature: {temperature}\n\n" + answer

            return direct_response_flag, answer
    else:
        direct_response_flag = True
        return direct_response_flag, "잘못된 RAG 유형 플래그입니다."

# >>>>> [끝] LCEL 및 메인 프로그램 영역


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message, unsafe_allow_html=True)    # markdown 형식의 답변 표시 가능함!
    if save:
        save_message(message, role)


# 기존의 대화 내용 전체를 다시 표시하기 위한 함수
def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,     # 다시 표시할 때는 session_state에 저장하지 않음. (여기서도 저장하면 중복 저장됨) 
        )


# def format_docs(docs):
#     return "\n\n".join(document.page_content for document in docs)


# -----

st.set_page_config(
    page_title="챗봇 - 한입해!",
    page_icon="📃",
)

st.markdown(
    """
    <div style='text-align:center;'>
    <h1 style='padding-bottom:1em'>🎓입시 정보 고민? <span style='color:orange;'>한입해!😋</span></h1>
    </div>
    <div style='text-align:center; padding-bottom:2em;'>
    <p>
    ❓ 다음과 같이 질문해 보세요~<br><br>
    건국대 국어국문학과 <span style="color:orange;">입시결과</span> 알려줘.<br>
    건국대 국어국문학과 <span style="color:orange;">경쟁률</span> 알려줘.<br>
    건국대 2025 <span style="color:orange;">모집인원</span> 알려줘.<br>
    서강대 학생부교과 <span style="color:orange;">지원자격</span> 알려줘.<br> 
    서강대 <span style="color:orange;">학생부종합</span>에 대해 자세히 알려줘.<br> 
    서강대 2025 수시 경쟁률 <span style="color:orange;">뉴스에서 검색해줘.</span><br><br>
    이 외에도 <span style="color:orange;">'변경사항', '전형일정', '전형 종류', '전형별 반영비율', '수능최저', '생기부 반영 방법', <br>'평가 방법', '지원 전략', '출제 유형'</span> 등의 키워드를 포함해 마음껏 질문해 보세요^^
    </p>
    </div>
    """,
    unsafe_allow_html=True
)

# st.markdown(
#     """<br><br>
# 다음과 같이 질문해 보세요~
# - 건국대 국어국문학과 <span style="color:orange;">입시결과</span> 알려줘.
# - 건국대 국어국문학과 <span style="color:orange;">경쟁률</span> 알려줘.
# - 건국대 2025 <span style="color:orange;">모집인원</span> 알려줘.
# - 서강대 학생부교과 <span style="color:orange;">지원자격</span> 알려줘. 
# - 서강대 <span style="color:orange;">학생부종합</span>에 대해 자세히 알려줘. 
# - 이 외에도 <span style="color:orange;">'변경사항', '전형일정', '전형 종류', '전형별 반영비율', '수능최저', '생기부 반영 방법', '평가 방법', '지원 전략', '출제 유형', '기사 검색'</span> 등의 키워드를 포함해 마음껏 질문해 보세요^^
# """, 
#     unsafe_allow_html=True
# )


# ===== RAG 사전 작업 영역 =====
# st.cache_data 데코레이터에 의해 최초 한 번만 실행됨. 
# 그 다음 부터는 사전에 캐싱된 code_rag, vectorstore, code_kiwi, db, news_db 객체를 계속 재사용하게 됨!!
set_env()   # 환경 설정. load_dotenv()로 API Key 로드
code_rag, vectorstore, chat_prompt = prepare_rag()
db, news_db = get_databases()
code_db_rag, sql_db_prompt, answer_db_prompt = prepare_db_rag()
code_web_rag, sql_web_prompt, answer_web_prompt = prepare_web_rag()
code_kiwi = get_codekiwi()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

send_message("안녕하세요! 저는 한입해 서비스를 담당하고 있는 <span style='color:green;'>꿀꺽</span>이예요.<br>입시에 관한 것이라면 무엇이든 질문해 주세요^^", "ai", save=False)

paint_history()

message = st.chat_input("입시에 관한 궁금증을 마음껏 질문해 보세요..")

if message:
    send_message(message, "human")

    with st.chat_message("ai"):
        # [참고] model_str, temperature 옵션은 문서 기반 RAG에만 적용됨. 
        direct_response_flag, response = execute_conditional_rag(vectorstore, message, model_str="GPT-4o", temperature=0.1)
        # direct_response_flag, response = execute_conditional_rag(vectorstore, message, model_str="Claude 3.5 Sonnet", temperature=0.1)

        if direct_response_flag:    # 랭체인 거치지 않고 곧바로 답변하는 경우
            st.markdown(response)
            save_message(response, "ai")

    # with st.sidebar:
    #     st.write(st.session_state["messages"])
