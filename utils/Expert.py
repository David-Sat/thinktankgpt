from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

from utils.Worker import Worker



class Expert(Worker):
    def __init__(self, model, role):
        super().__init__(model=model, role=role)




    def test_expert(self, input_text, stream_handler):
        list_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You're a helpful assistant."),
                ("human", "Return a prime number"),
            ]
        )
        chain = (
            list_prompt
            | self.model
            | StrOutputParser()
        )
        config = {
            "callbacks": [stream_handler]
        }
        return chain.invoke(input={}, config=config)