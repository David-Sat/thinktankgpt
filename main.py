import streamlit as st

from utils.StreamHandler import StreamHandler
from utils.Debate import Debate

def get_openai_api_key():
    if "openai_api_key" not in st.session_state:
        if "openai_api_key" in st.secrets:
            st.session_state["openai_api_key"] = st.secrets.openai_api_key
        else:
            st.session_state["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password")
    if not st.session_state["openai_api_key"]:
        st.info("Enter an OpenAI API Key to continue")
        st.stop()

def initialize():
    st.session_state["initialized"] = True
    get_openai_api_key()
    st.session_state["experts"] = []
    st.session_state["debate"] = Debate(openai_api_key=st.session_state["openai_api_key"], model_name="gpt-3.5-turbo")


st.markdown(
    """
    <style>
    .stButton > button {
        width: 100%;
        height: auto;
    }
    
    /* Apply margin-top only for non-mobile views */
    @media (min-width: 768px) {
        [data-testid="stFormSubmitButton"] > button {
            margin-top: 28px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("ThinkTankGPT")

form = st.form(key="form_settings")
topicCol, buttonCol = form.columns([4,1])

topic = topicCol.text_input(
    "Enter your topic of debate",
    key="topic",
)


expander = form.expander("Customize debate")
number_experts = expander.slider(
    'Number of experts:',
    min_value=2,      
    max_value=6, 
    value=3,
    step=1
)

options = ["Strongly For", "For", "Neutral", "Against", "Strongly Against"]
stance = expander.select_slider("Stance of the experts", options=options, value="Neutral")
submitted = buttonCol.form_submit_button(label="Submit", on_click=initialize)

chat = st.container()

if "debate" in st.session_state:
    for msg in st.session_state.debate.debate_history:
        chat.chat_message(name=msg["role"], avatar=msg["avatar"]).write(msg["content"])


def debate_round():
    for expert in st.session_state["experts"]:
        with chat.chat_message(name=expert.expert_instruction["role"], avatar=expert.expert_instruction["avatar"]):
            response = expert.generate_argument(st.session_state.debate, StreamHandler(st.empty()))
            st.session_state.debate.add_message(role=expert.expert_instruction["role"], avatar=expert.expert_instruction["avatar"], content=response)

if submitted and topic.strip() != "":
    st.session_state["debate"].initialize_new_debate(topic=topic, num_experts=number_experts)
    st.session_state["experts"] = st.session_state["debate"].get_experts()

    debate_round()

if "initialized" in st.session_state and st.session_state["initialized"]:
    if input := st.chat_input(placeholder="Participate in the debate"):
        user_prompt = input.replace("Human: ", "")
        st.session_state.debate.add_message(role="user", content=user_prompt)
        chat.chat_message("user").write(user_prompt)

        debate_round()

    st.button("Continue debate", on_click=debate_round)

