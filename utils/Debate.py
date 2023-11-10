from langchain.schema import ChatMessage


class Debate():
    def __init__(self):
        self.topic = None
        self.debate_history = []
        self.memory = []

    def add_message(self, role, content, avatar=None):
        self.debate_history.append({"role": role, "avatar": avatar, "content": content})
        role = "user" if role == "user" else "assistant"
        self.memory.append(ChatMessage(role=role, content=content))


    def set_topic(self, topic):
        self.topic = topic