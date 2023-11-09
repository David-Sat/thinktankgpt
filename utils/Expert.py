from operator import itemgetter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import StrOutputParser, ChatMessage

from langchain.memory import ConversationBufferMemory

from utils.Worker import Worker



class Expert(Worker):
    def __init__(self, model, expert_instruction):
        super().__init__(model=model)
        self.expert_instruction = expert_instruction
        self.system_prompts = self.config["expert"]['system_prompts']
        self.examples = self.config["expert"]['examples']


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
    
    def generate_argument(self, debate_history, topic, stream_handler):
        system_prompt = self.system_prompts["system1"].replace("##debate_topic##", topic)
        system_prompt += "\n" + self.expert_instruction["instructions"]

        messages = [ChatMessage(role="system", content=system_prompt)]
        #messages.append(debate_history)

        debate_prompt = ChatPromptTemplate.from_messages(messages)

        chain = (
            debate_prompt
            | self.model
            | StrOutputParser()
        )

        config = {
            "callbacks": [stream_handler]
        }
        return chain.invoke(input={}, config=config)
