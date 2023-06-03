from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import openai


def main():
    load_dotenv()
    st.set_page_config(page_title = "PDF Reader")
    st.title("PDF Reader")
    st.header("Created by Leo.")

    pdf = st.file_uploader("Upload your PDF file", type = ["pdf"])
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()  
        
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
        user_question = st.text_input("Enter your question about the pdf")
        if user_question:
            docs = document.similarity_search(user_question, k = 3)
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")


            response = chain.run(input_documents = docs, question = user_question)

            st.write(response)

        

if __name__ == '__main__':
    main()