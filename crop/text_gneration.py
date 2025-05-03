import os
import langchain
import torch
from InstructorEmbedding import INSTRUCTOR
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.globals import set_llm_cache
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
from appcopy import get_pdf_text, get_text_chunks, get_pdf_docs

#yolo model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import re
import os
import matplotlib.pyplot as plt
import cv2
import easyocr
from pylab import rcParams
from IPython.display import Image


def process_image_and_extract_text(var="202404161152074646.png"):
    output = subprocess.run(['yolo', 'segment', 'predict', f'model=./runs/best.pt', f'source="./infrence/{var}"', 'save=True'], capture_output=True, text=True)

    # Extract the predict number from the output
    matches = re.search(r'predict(\d+)', output.stdout)
    if matches:
        predict_number = int(matches.group(1)) 
        result_path = f"runs/segment/predict{predict_number}"
    else:
        print("Predict number not found in output.")
        result_path = None

    filename = os.path.basename(var)

    if result_path and filename:
        segmented_image_path = os.path.join(result_path, filename)
        print("Segmented image path:", segmented_image_path)
    else:
        print("Unable to construct segmented image path.")

    reader = easyocr.Reader(['en'])
    output = reader.readtext(segmented_image_path)
    result = ["".join(filter(str.isalpha, item[1])) for item in output if item[1].split()[0] in {'Rust', 'Miner', 'Phoma'}]

    return result

api_key = "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"
llm = ChatOpenAI(openai_api_key=os.environ["OPEN_API_KEY"], temperature=0, max_tokens=150)
langchain.llm_cache = InMemoryCache()

embeddings = OpenAIEmbeddings(openai_api_key=api_key)
vectordb_file_path = "croma_index"
persist_directory = 'db'

pdf_docs = get_pdf_docs()
def vector_db():
    # Extract text from PDF files
    raw_text = get_pdf_text(pdf_docs)

    # Split text into chunks
    text_chunks = get_text_chunks(raw_text)

    # Create vector store
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

def get_image():
    var="202404161152074646.png"
    return var

if __name__ == "__main__":
    # Create vector store from PDF files
    vectordb = vector_db()
    ressult=process_image_and_extract_text(var='202404161152074646.png')
    # Create the question-answering chain
    chain = chain(vectordb)

    # Ask a question
    # result_text = chain(f"what is {ressult} in leaves")['result']
    result_text = chain(f"what is {ressult} in coffee plants or fungal plant pathogen or that causes dieback")['result']
    last_full_stop_index = result_text.rfind('.')
    if last_full_stop_index != -1:
        result_text = result_text[:last_full_stop_index + 1]
    print(result_text)
    