from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

os.environ['OPENAI_API_KEY'] = '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'
load_dotenv()

app = Flask(__name__)

# In-memory storage for PDF texts
pdf_texts = []

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

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
def get_pdf_docs():
    pdf_docs = ['pdf/ManualRoyaInglesCATIE.pdf']
    return pdf_docs


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
    app.run(debug=True)