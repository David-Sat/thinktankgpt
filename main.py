from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate  
from langchain.schema import ChatMessage

import streamlit as st

from utils.StreamHandler import StreamHandler
from utils.WorkerManager import WorkerManager
from utils.Debate import Debate

def initialize():
    st.session_state["initialized"] = True
    # Get an OpenAI API Key before continuing
    if "openai_api_key" not in st.session_state:
        if "openai_api_key" in st.secrets:
            st.session_state["openai_api_key"] = st.secrets.openai_api_key
        else:
            st.session_state["openai_api_key"] = st.sidebar.text_input("OpenAI API Key", type="password")
    if not st.session_state["openai_api_key"]:
        st.info("Enter an OpenAI API Key to continue")
        st.stop()

    st.session_state["worker_manager"] = WorkerManager(openai_api_key=st.session_state["openai_api_key"], model_name="gpt-3.5-turbo")
    st.session_state["experts"] = []
    
st.markdown(
    """
<style>
/* Styling the button */
.stButton > button {
    width: 100%;
    height: auto;
    margin-top: 28px; 
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("ThinkTankGPT")
header_title = ""
st.header(header_title)

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
submitted = buttonCol.form_submit_button(label="Submit")

if "initialized" not in st.session_state:
    initialize()

if "messages" not in st.session_state:
    st.session_state["messages"] = Debate()

for msg in st.session_state.messages.debate_history:
    st.chat_message(msg.role).write(msg.content)

worker_manager = st.session_state["worker_manager"]

def debate_round():
    for expert in st.session_state["experts"]:
        with st.chat_message("expert"):
            response = expert.generate_argument(st.session_state.messages, StreamHandler(st.empty()))
            st.session_state.messages.add_message(role=expert.expert_instruction["role"], content=response)

if submitted and topic.strip() != "":
    st.session_state.messages.add_message(role="user", content=topic)
    st.session_state.messages.set_topic(topic)
    st.chat_message("user").write(topic)
    st.session_state["experts"] = worker_manager.create_experts(num_experts=number_experts, topic=topic)

    debate_round()


if input := st.chat_input(placeholder="Participate in the debate"):
    user_prompt = input.replace("Human: ", "")
    st.session_state.messages.add_message(role="user", content=user_prompt)
    #st.session_state.messages.append(ChatMessage(role="user", content=user_prompt))
    st.chat_message("user").write(user_prompt)
    #experts = worker_manager.create_experts(num_experts=number_experts, topic=user_prompt)

    debate_round()

    

