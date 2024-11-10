from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st 
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.6,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def load_file(file_path):
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    elif file_path.endswith(".pdf"):
        pdf_reader = PdfReader(file_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    else:
        raise ValueError("Unsupported file format. Please use a .txt or .pdf file.")
    return text

file_path = input("Please provide the path to the file you want to chat with: ")

try:
    file_content = load_file(file_path)
    print("File loaded successfully. Starting chat...")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

print("You can now start asking questions about the file content.")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Exiting chat. Goodbye!")
        break

    messages = [
        ("system", "You are a helpful assistant that answers questions based on the uploaded file content."),
        ("context", file_content),  # Load the file content as context
        ("human", query)
    ]
    
    try:
        ai_msg = llm.invoke(messages)
        print("AI:", ai_msg.content)
    except Exception as e:
        print(f"An error occurred: {e}")