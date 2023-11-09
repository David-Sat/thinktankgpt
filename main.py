from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate  
from langchain.schema import ChatMessage

from utils.StreamHandler import StreamHandler

import streamlit as st

from utils.WorkerManager import WorkerManager

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
    # Instantiate the workers
    conv_template = ChatPromptTemplate.from_template("{topic}")
    #st.session_state["coordinator"] = Coordinator(openai_api_key=st.session_state["openai_api_key"])
    
    #st.session_state["worker1"] = Expert(openai_api_key=st.session_state["openai_api_key"], role="expert", conv_template=conv_template)


if "initialized" not in st.session_state:
    initialize()


value = st.sidebar.slider(
    'Number of experts:',
    min_value=2,      
    max_value=6, 
    value=3,
    step=1
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

worker_manager = st.session_state["worker_manager"]

if input := st.chat_input(placeholder="Enter your topic of debate"):
    user_prompt = input.replace("Human: ", "")
    st.session_state.messages.append(ChatMessage(role="user", content=user_prompt))
    st.chat_message("user").write(user_prompt)
    worker_manager.create_experts(num_experts=value, topic=user_prompt)

    with st.chat_message("assistant2"):
        response = worker_manager.invoke_experts(input, StreamHandler(st.empty()))
        pass
        #response = coordinator.generate_expert_instructions()
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))

    #with st.chat_message("assistant"):
        #response = current_worker.get_response(input, st.empty(), st.session_state.messages)
        #st.session_state.messages.append(ChatMessage(role="assistant", content=response))

