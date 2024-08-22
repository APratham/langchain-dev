from transformers import T5Tokenizer, TFT5ForConditionalGeneration
from transformers import pipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

print("Hugging Face Token:", os.getenv("HUGGINGFACE_HUB_TOKEN"))


model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False, clean_up_tokenization_spaces=True)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

prompt = PromptTemplate(
    input_variables=["log"],
    template="Please explain the following log entry to me in two or three lines: {log}"
)

def generate_response(input_text):
    if isinstance(input_text, str):
        text = input_text
    elif hasattr(input_text, 'text'):
        text = input_text.text
    else:
        raise ValueError("Input to the tokenizer must be a string or StringPromptValue.")

    inputs = tokenizer(text, return_tensors="tf")
    print(f"Inputs: {inputs}")
    
    outputs = model.generate(**inputs, max_length=500, max_new_tokens=100)  # Increase max_length
    print(f"Raw Outputs: {outputs}")
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

llm_runnable = RunnableLambda(generate_response)

runnable_sequence = prompt | llm_runnable

log_entry = "ERROR: 2024-08-19 10:33:00 - Unable to connect to the database."

explanation = runnable_sequence.invoke({"log": log_entry})
print("Explanation:", explanation)
