from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from utils.StreamHandler import StreamHandler

class Worker:
    def __init__(self, model, role):
        self.model = model
        self.role = role

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
    
    def build_chat_history(self, messages):
        pass



class Expert(Worker):
    def __init__(self, model, role):
        super().__init__(model=model, role=role)