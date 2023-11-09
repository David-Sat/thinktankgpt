from langchain.chat_models import ChatOpenAI
from utils.Coordinator import Coordinator
from utils.Worker import Worker
from utils.Expert import Expert

class WorkerManager:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.coordinator = None
        self.experts = []

    def create_experts(self, num_experts, topic):
        coordinator_model = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name)
        self.coordinator = Coordinator(coordinator_model, num_experts, topic)

        expert_list = self.coordinator.generate_expert_instructions()

        for expert in expert_list:
            expert_model = ChatOpenAI(openai_api_key=self.openai_api_key, model_name=self.model_name, streaming=True)
            self.experts.append(Expert(expert_model, expert["role"]))

        
            
    def invoke_experts(self, input_text, stream_handler):
        response = self.experts[0].test_expert("test", stream_handler)
        print(response)
        return response