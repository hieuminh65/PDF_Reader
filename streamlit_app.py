from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

with st.sidebar:
    st.set_page_config(page_title="PDF Reader", page_icon="📚", layout="wide")
    st.markdown('''
    ## How this works
    When you upload a document, it gets split into smaller sections and stored in a special kind of database called a vector index. This type of database allows for semantic search and retrieval, which means it can find related information even if the individual words aren't exact matches. 

    When you ask a question, the model searches through these document sections using the vector index to find relevant information and ultimately provide an answer.

    ## About me
    Check out my [Web](https://mywebleo.com)
    ''')
    



def main():
    st.title("PDF Reader")
    load_dotenv()
    pdf = st.file_uploader("Upload your PDF file", type = ["pdf"])
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()  
        
        #split text into chunks
        text_splitter = CharacterTextSplitter(
            separator= "\n",
            chunk_size = 1000,
            chunk_overlap = 100,
            length_function = len 
        )
        chunks = text_splitter.split_text(text)

        #create embeddings
        embeddings = OpenAIEmbeddings()
        document = FAISS.from_texts(chunks, embeddings)
        
        #search
        user_question = st.text_input("Enter your question about the pdf: ")
        if user_question:
            docs = document.similarity_search(user_question, k = 3)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")


            response = chain.run(input_documents = docs, question = user_question)

            st.write(response)

        

if __name__ == '__main__':
    main()