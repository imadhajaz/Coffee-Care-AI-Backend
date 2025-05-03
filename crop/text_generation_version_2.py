from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.cache import InMemoryCache
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.globals import set_llm_cache
set_llm_cache(InMemoryCache())
load_dotenv()
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from random import randint
from datetime import datetime
import os
import easyocr
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import langchain
import re

os.environ['OPENAI_API_KEY']='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'
def get_pdf_docs():
    pdf_docs = ['pdf/ManualRoyaInglesCATIE.pdf']
    return pdf_docs

def process_image_and_extract_text(var):
    output = subprocess.run(['yolo', 'segment', 'predict', f'model=./runs/best.pt', f'source="{var}"', 'save=True'], capture_output=True, text=True)

    # Extract the predict number from the output
    matches = re.search(r'predict(\d+)', output.stdout)
    print(matches)
    if matches:
        predict_number = int(matches.group(1)) 
        result_path = f"runs/segment/predict{predict_number}"
    else:
        print("Predict number not found in output.")
        result_path = None

    filename = os.path.basename(var)
    if result_path and filename:
        print(filename)
        segmented_image_path = str(result_path) + '/' + str(filename)
        print("Segmented image path:", segmented_image_path)
    else:
        print("Unable to construct segmented image path.")
    
    reader = easyocr.Reader(['en'])
    output = reader.readtext(segmented_image_path)
    result = set(["".join(filter(str.isalpha, item[1])) for item in output if item[1].split()[0] in {'Rust', 'Miner', 'Phoma'}])
    return result, segmented_image_path

api_key = "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"
llm = ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"], temperature=0, max_tokens=150)
langchain.llm_cache = InMemoryCache()

embeddings = OpenAIEmbeddings(openai_api_key="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")
vectordb_file_path = "croma_index"
persist_directory = 'db'

pdf_docs = get_pdf_docs()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def vector_db():
       
    raw_text = get_pdf_text(pdf_docs)

    text_chunks = get_text_chunks(raw_text)

   
    vectordb = Chroma.from_texts(texts=text_chunks, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    return vectordb

def chain(vectordb):
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """You are an Expert on Horticulture. Generate a description based on the context only. Try to provide as much text as possible from the "response" section in the source documents.
    CONTEXT: {context}
    QUESTION: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=['CONTEXT', 'QUESTION']
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain

app = Flask(__name__, static_url_path="/runs", static_folder="runs")

vector_db_var = vector_db()

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files['image_file']
    var = "infrence/" + str(datetime.now().strftime('%Y%m%d%H%M%S')) + str(randint(0, 10000)) + '.png'
    image_file.save(var)
    chain_var = chain(vector_db_var)
    results = process_image_and_extract_text(var)
    
    # result_text = chain_var(f"what is {results[0]} in leaves")['result']
    result_text = chain_var(f"what is {results[0]} in coffee plants or fungal plant pathogen or that causes dieback")['result']
    disease_detect=f"Disease Detected:{results[0]}"
    last_full_stop_index = result_text.rfind('.')
    if last_full_stop_index != -1:
        result_text = result_text[:last_full_stop_index + 1]
    return jsonify({'Description':result_text,'disease':disease_detect, 'image_path': results[1]})

# In-memory storage for PDF texts
pdf_texts = []

# Function to extract text from PDF file

# Function to split text into chunks

# Function to create a vector store
def create_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    vectorstore = Chroma.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational chain
def create_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.5)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

# API endpoint to add PDF
@app.route('/add_pdf', methods=['POST'])
def add_pdf():
    global pdf_texts, vectorstore, conversation_chain
    pdf_files = request.files.getlist('pdf_files')
    if not pdf_files:
        return jsonify({'error': 'No PDF files provided'}), 400
    for pdf in pdf_files:
        raw_text = get_pdf_text([pdf])
        pdf_texts.append(raw_text)
    text_chunks = []
    for text in pdf_texts:
        chunks = get_text_chunks(text)
        text_chunks.extend(chunks)
    vectorstore = create_vectorstore(text_chunks)
    conversation_chain = create_conversation_chain(vectorstore)
    return jsonify({'message': 'PDFs added successfully'})

# API endpoint to ask question
@app.route('/ask_question', methods=['POST'])
def ask_question():
    global vectorstore, conversation_chain
    if vectorstore is None or conversation_chain is None:
        return jsonify({'error': 'No PDFs uploaded yet'}), 400
    question = request.form.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    response = conversation_chain.run(question)
    return jsonify({'result': response})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)