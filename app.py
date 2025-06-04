from flask import Flask, render_template, request, redirect, jsonify
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.helper import load_pdf_file, text_split, download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_together import Together
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

embeddings = download_huggingface_embeddings()

index_name = "medicalbot"

docsearch= PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name 
)

retriever= docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2",  # or another available model
    temperature=0.4,
    max_tokens=500,
)

prompt= ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain= create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/ask', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    input= msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response: ",response["answer"])
    return str(response["answer"])


if __name__ == '__main__':  
    app.run(host='0.0.0', port=8080, debug=True)  # Set debug=True for development
  