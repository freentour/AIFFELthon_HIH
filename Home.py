import streamlit as st
# from datetime import datetime

# 웹페이지 설정
st.set_page_config(
    page_title="Home - 수시 전문 챗봇 한입해!",
    page_icon="🎓"      # [참고] 이모지 입력 단축키: 윈도우 + .
)

# # 사이드바 영역
# with st.sidebar:
#     st.title("Sidebar Title")
#     st.text_input("Input Text Test")

# # 탭 생성
# tab1, tab2, tab3 = st.tabs(['A', 'B', 'C'])

# with tab1:
#     st.write('a')

# with tab2:
#     st.write('b')

# with tab3:
#     st.write('c')


# # CSS를 사용하여 fixed bottom 스타일 정의
# st.markdown(
#     """
#     <style>
#     /* 메인 컨텐츠를 위한 여백 확보 */
#     /*
#     .main {
#         padding-bottom: 100px; /* 하단 요소의 높이만큼 여백 확보 */
#     }
#     */
#     .orange {
#         color: rgb(255,0,0);
#     }
    
#     /* 하단 고정 요소를 위한 스타일 */
#     .fixed-bottom {
#         position: fixed;
#         bottom: 0;
#         left: 0;
#         right: 0;
#         background-color: white;
#         padding: 20px;
#         z-index: 999;
#         border-top: 1px solid #ddd;
#     }
    
#     /* Streamlit 기본 패딩 조정 */
#     .block-container {
#         padding-bottom: 100px;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# 메인 콘텐츠 영역
# st.title("🎉 입시 정보 고민? 한.입.해!")
# st.subheader("(한입해 : 한방에 입시 해결)")

# # 페이지 내에서 일부만 변경되어도(사용자 입력의 변화 포함해서) 전체 폐이지가 모두 refresh 되는 것을 확인하기 위해 시간을 삽입해 봄.
# today = datetime.today().strftime("%H:%M:%S")
# st.header(today)

# st.divider()

st.markdown(
    """
    <div style='text-align:center;'>
    <h1 style='padding-bottom:0em'>🎓입시 정보 고민? <span style='color:orange;'>한입해!😋</span></h1>
    <h3 style='color:grey; margin-bottom:0.5em'>(한입해 : 한방에 입시 해결)</h3>
    <h3 style='margin-bottom:0.5em'>안녕하세요!<br><span style="color:orange;">수시 정보 전문 챗봇</span><br>한입해 서비스에 오신 것을 환영합니다.</h3>
    <p style='color:grey; margin-bottom:1.5em;'>
    💦 매년 바뀌고 대학마다 다른 입시(수시) 정보, 도대체 어디서부터 시작해야할지 막막하신가요?<br>
    💭 혹시 이런 질문하면 너무 멍청해 보일지 걱정되나요?<br>
    💥 카더라 통신은 그만! 신뢰할만한 정보 위주로만 찾아볼 순 없을까?<br>
    </p>
    <p style='margin-bottom:1.5em;'>
    24시간 아무 때나 질문해도 싫은 기색 하나 없이 친절하게 대답해주는<br>
    <span style='color:orange;'>수시 정보 전문 챗봇</span> 한입해 서비스로 마음껏 질문해 보세요!!<br>
    </p>
    </div>
    <hr>
    <div>
    [참고 및 주의사항]<br>
    💡 한입해 서비스는 수시 정보 전문 챗봇입니다.<br>
    💡 다음에 올 때는 정시 정보도 열심히 공부해서 올게요!💪<br>
    💡 한입해 서비스는 대학교 입시 요강을 비롯한 신뢰할 수 있는 문서들 위주로만 답변하도록 제작되었습니다.<br>
    📢 그럼에도 불구하고 AI(인공지능)는 언제든 실수하거나 잘못된 정보를 제공할 가능성이 있습니다.😅<br>
    📢 따라서, <span style="color:orange;">중요한 내용은 항상 대학교에서 발간된 공식 문서(입시 요강)를 통해 크로스 체크해 주시거나, 진학 지도 경험이 많은 학교 선생님과 상담하시는 것을 적극 추천</span>합니다.<br>
    </div>
    """,
    unsafe_allow_html=True
)

# st.divider()


# model = st.selectbox(
#     "AI 모델 선택",
#     (
#         "GPT-3-turbo",
#         "GPT-4o-mini",
#         "GPT-4o"
#     ),
#     index=None,
#     placeholder="사용하려는 모델을 선택해 주세요."
# )

# if model == "GPT-3-turbo":
#     name = st.text_input("당신의 이름은?")
#     st.write(name)
# elif model is None:
#     pass
# else:
#     # slider 메소드의 기본 인자 순서 : 레이블, 최소값, 최대값, 기본값, 간격)
#     value = st.slider("temperature", 0.0, 2.0, 1.0, 0.1)
#     st.write(value)