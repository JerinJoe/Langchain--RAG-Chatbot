import streamlit as st
import torch
from transformers import GPT2Tokenizer
from langchain.llms.base import LLM
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field

# Set device for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model_path = r"Path_to_your_model.pt"
model = torch.load(model_path, map_location=device)
model.eval()  # Set to evaluation mode

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Define a custom LangChain LLM class
class CustomGPTModel(LLM):
    local_model: torch.nn.Module = Field(...)  
    tokenizer: GPT2Tokenizer = Field(...)      

    def __init__(self, model, tokenizer):
        super().__init__()  # Ensure base LLM is initialized
        self.local_model = model  
        self.tokenizer = tokenizer

    def _call(self, prompt: str, stop=None):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.local_model.generate(**inputs, max_length=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    @property
    def _llm_type(self):
        return "custom_gpt"

# Initialize the custom LLM
custom_llm = CustomGPTModel(model=model, tokenizer=tokenizer)

# Function to create a prompt
def create_prompt(question):
    # Only use the core topic for the prompt without the question structure
    return f" {question}"

# Streamlit app
st.title("Chatbot")
st.write("Ask me anything!")

# User input
user_question = st.text_input("Your Question:")

if st.button("Search"):
    if user_question:
        # Create the prompt
        prompt_text = create_prompt(user_question)
        
        # Get the model's response
        answer = custom_llm(prompt_text)
        # answer= answer.split('|')[-1]
        st.write(answer)
    else:
        st.write("Please enter a question.")
