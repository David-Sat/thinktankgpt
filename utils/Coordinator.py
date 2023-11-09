
from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from utils.Worker import Worker
from langchain.schema import StrOutputParser

import json
from pathlib import Path

example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )

class Coordinator(Worker):
    def __init__(self, model, num_experts, topic):
        super().__init__(model=model, role="coordinator")
        self.num_experts = num_experts
        self.topic = topic
        self.load_config()

    def load_config(self):
        config_path = Path(__file__).resolve().parent.parent / 'configs' / 'config.json'
        
        with config_path.open('r', encoding='utf-8') as f:
            config = json.load(f)

        self.coordinator_system1 = config['coordinator_systems']['system1']
        self.coordinator_system2 = config['coordinator_systems']['system2']
        self.examples1 = config['examples']['examples1']
        self.examples2 = config['examples']['examples2']


    def generate_expert_instructions(self):
            """
            Generates expert instructions for the given topic and number of experts.

            Returns:
            expert_list (list): A list of dictionaries containing expert information and instructions.
            """
            list_prompt = self.create_list_prompt(self.num_experts, self.topic, example_prompt)
            chain1 = (
                list_prompt
                | self.model
                | StrOutputParser()
            )

            expert_list = self.process_expert_list(chain1.invoke({}))
            batch_inputs = []
            for expert in expert_list:
                batch_inputs.append({"role": expert["role"], "stance": expert["stance"], "topic": self.topic})
            
            expert_instruction_prompt = self.create_expert_instruction_prompt(example_prompt)
            chain2 = (
                expert_instruction_prompt
                | self.model
                | StrOutputParser()
            )

            expert_instructions = chain2.batch(batch_inputs)

            for i, expert in enumerate(expert_list):
                expert["instructions"] = expert_instructions[i]

            return expert_list

    def process_expert_list(self, expert_list_raw):
        expert_list = expert_list_raw.split("; ")
        result_list = []
        for expert in expert_list:
            role, stance = expert.split(", ")
            result_list.append({"role": role, "stance": stance})
        return result_list


    def create_list_prompt(self, num_experts, topic, example_prompt):
            """
            Creates a chat prompt template for selecting a list of experts for a given topic.

            Args:
                num_experts (int): The number of experts to select.
                topic (str): The topic for which experts are being selected.
                example_prompt (str): The prompt to use for the few-shot examples.

            Returns:
                ChatPromptTemplate: The chat prompt template for selecting experts.
            """
            coordinator_list_instruction = self.coordinator_system1.replace("##num_workers##", str(num_experts))
            few_shot_prompt_list_experts = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=self.examples1,
            )
            human_query = f"{num_experts} experts for the topic: {topic}"
            list_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", coordinator_list_instruction),
                    few_shot_prompt_list_experts,
                    ("human", human_query),
                ]
            )
            return list_prompt


    def create_expert_instruction_prompt(self, example_prompt):
            """
            Creates a chat prompt template for expert instruction.

            Args:
                example_prompt (str): The example prompt to use in the chat prompt.

            Returns:
                ChatPromptTemplate: The chat prompt template for expert instruction.
            """
            few_shot_prompt_instruct_expert = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=self.examples2,
            )
            expert_instruction_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.coordinator_system2),
                    few_shot_prompt_instruct_expert,
                    ("human", "role: {role}; stance: {stance}; topic: {topic}"),
                ]
            )
            return expert_instruction_prompt
