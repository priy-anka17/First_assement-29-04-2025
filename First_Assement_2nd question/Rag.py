from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.schema import Document
import torch
from huggingface_hub import login, whoami
import os


def data_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        document = file.read()
    return [Document(page_content=document)]

def datasplitter(document, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(document)
    return chunks

def create_embedding():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

def create_vectordb(chunks, embedding, persist_directory='./chromadb'):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb


def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever()

    model_id = "google/gemma-3-1b-it"  # Corrected model ID
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        device=device
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

def main(file_path):
    document = data_loader(file_path)
    chunks = datasplitter(document)
    embedding = create_embedding()
    vectordb = create_vectordb(chunks, embedding)
    qa_chain = create_qa_chain(vectordb)

    return qa_chain

# Log in to Hugging Face Hub
HUGGINGFACE_API_TOKEN = "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"  # Replace with your actual token
try:
    login(token=HUGGINGFACE_API_TOKEN)
    user_info = whoami()
    print(f"Successfully logged in as: {user_info['name']}")  # Verify login
except Exception as e:
    print(f"Login failed: {e}")

Query = "Tell me the executive summary of this document"
file_path = r"C:\Users\priya\Desktop\FIrst_assesment\BlackRock.txt"
qa_chain = main(file_path)

result = qa_chain({"query": Query})
print("Answer:", result['result'])