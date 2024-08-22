from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("HUGGINGFACE_HUB_TOKEN")

template = """
You are a friendly chatbot engaging in a conversation with a human.

Previous conversation:
{chat_history}

New human question:
{question} 
Response: 
"""

prompt = PromptTemplate.from_template(template) 

repo_id = "google/flan-t5-base"
llm = HuggingFaceHub( 
    repo_id=repo_id,
    model_kwargs={
        "temperature": 0.1,
        "max_length": 64
        },
)

memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

while True:
    query = input("User query: ")

    if query == "exit":
        break
    print(conversation({"question": query})['text'])