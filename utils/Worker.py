from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.StreamHandler import StreamHandler
from pathlib import Path
import json

class Worker:
    def __init__(self, model):
        self.model = model
        self.config = self.load_config()
        
    def load_config(self):
        config_path = Path(__file__).resolve().parent.parent / 'configs' / 'config.json'
        with config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def get_response(self, input_text, container, messages):
        model = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name, streaming=True, callbacks=[StreamHandler(container)])
        debate = ChatPromptTemplate.from_messages(
            messages
        )

        chain = (
            debate
            | model
        )

        response = ""
        for chunk in chain.stream({"topic": input_text}):
            response += chunk.content

        return response
    


