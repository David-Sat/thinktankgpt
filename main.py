import streamlit as st
from pathlib import Path
import json

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

def initialize_debate(start_new=True, debate_history=None, expert_instructions=None):
    get_openai_api_key()
    st.session_state["debate"] = Debate(openai_api_key=st.session_state["openai_api_key"], model_name="gpt-3.5-turbo")
    st.session_state["initialized"] = True
    st.session_state["experts"] = []

    if start_new:
        st.session_state.debate.initialize_new_debate(topic=topic, num_experts=number_experts, stance=stance)
    else:
        st.session_state.debate.initialize_existing_debate(topic=topic, debate_history=debate_history, expert_instructions=expert_instructions)

    st.session_state["experts"] = st.session_state.debate.get_experts()

def load_debate_configuration():
    config_path = Path(__file__).resolve().parent / 'configs' / 'suggestions.json'
    with config_path.open('r', encoding='utf-8') as f:
        config = json.load(f)
    st.session_state["suggestions"] = config["suggestions"]

def display_suggestions():
    suggestions = st.container()
    columns = suggestions.columns(4)
    for i, suggestion in enumerate(st.session_state["suggestions"][:4]):
        columns[i].button(suggestion["topic"], on_click=lambda: initialize_debate(start_new=False, debate_history=suggestion["debate_history"], expert_instructions=suggestion["expert_instructions"]))

def conduct_debate_round():
    default_avatar = "👤"
    for expert in st.session_state["experts"]:
        try:
            with chat.chat_message(name=expert.expert_instruction["role"], avatar=expert.expert_instruction["avatar"]):
                response = expert.generate_argument(st.session_state.debate, StreamHandler(st.empty()))
                st.session_state.debate.add_message(role=expert.expert_instruction["role"], avatar=expert.expert_instruction["avatar"], content=response)
        except Exception:
            with chat.chat_message(name=expert.expert_instruction["role"], avatar=default_avatar):
                response = expert.generate_argument(st.session_state.debate, StreamHandler(st.empty()))
                st.session_state.debate.add_message(role=expert.expert_instruction["role"], avatar=default_avatar, content=response)


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

# Settings and form
form = st.form(key="form_settings")
topicCol, buttonCol = form.columns([4,1])

topic = topicCol.text_input(
    "Enter your topic of debate",
    key="topic"
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

# Trigger initialization
submitted = buttonCol.form_submit_button(label="Submit")
if submitted and topic.strip():
    initialize_debate()

# Load and display suggestions
load_debate_configuration()
display_suggestions()

## Chat interface
chat = st.container()

if "debate" in st.session_state:
    for msg in st.session_state.debate.debate_history:
        chat.chat_message(name=msg["role"], avatar=msg["avatar"]).write(msg["content"])

if submitted and topic.strip() != "":
    st.session_state["debate"].initialize_new_debate(topic=topic, num_experts=number_experts, stance=stance)
    st.session_state["experts"] = st.session_state["debate"].get_experts()

    conduct_debate_round()

if "initialized" in st.session_state and st.session_state["initialized"]:
    expert_expander = form.expander("Generated Experts")
    for expert in st.session_state.experts:
        expert_expander.write(f"{expert.expert_instruction['avatar']} {expert.expert_instruction['role']}, {expert.expert_instruction['stance']}")
    if input := st.chat_input(placeholder="Participate in the debate"):
        user_prompt = input.replace("Human: ", "")
        st.session_state.debate.add_message(role="user", content=user_prompt)
        chat.chat_message("user").write(user_prompt)

        conduct_debate_round()

    st.button("Continue debate", on_click=conduct_debate_round)

