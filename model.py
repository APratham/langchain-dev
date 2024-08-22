from langchain_core import prompts
from langchain_openai import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

template = """
You are an expert data scientist with expertise in building deep learning models.
Explain the concept of {concept} to a beginner in a few sentences.
"""

if api_key is None:
    raise ValueError("API key not found. Please check your .env file.")
else: print(api_key + "\n")

llm= OpenAI(model_name="gpt-3.5-turbo-instruct", api_key=api_key)

prompt = prompts.PromptTemplate(
    input_variables=["concept"],
    template=template
)   

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    )

embeddings = OpenAIEmbeddings()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
    )

pc.create_index(
    name="concept-explanations",
    dimension=1536, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

index_name = "concept-explanations"
query = "What is regularisation?"   

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("regularisation"))

second_prompt = prompts.PromptTemplate(
    input_variables=["ml_concept"],
    template="Turn the concept description of {ml_concept} into simpler terms. Explain it to me like I'm five, but in 500 words"
)

chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

explanation = overall_chain.run("regularisation")

texts = text_splitter.create_documents([explanation])


print(texts[0].page_content)

query_result = embeddings.embed_query(texts[0].page_content)
print(query_result)
print("Query vector dimension:", len(query_result))  # Print the dimension of the query vector


search = PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
result = search.similarity_search(query)
print(result)
