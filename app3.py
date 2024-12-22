# Streamlit imports
import streamlit as st

# Necessary LangChain and Hugging Face imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from operator import itemgetter
import os
from dotenv import load_dotenv

hf_token = os.getenv('HF1_TOKEN')

# LangSmith documentation loader and vector store
loader = PyPDFLoader(r'C:\Users\Yasir\Desktop\RAG\National Ai policy.pdf')
docs = loader.load()

embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20).split_documents(docs)
vectordb = FAISS.from_documents(documents, embed_model)
retriever = vectordb.as_retriever()

# Set up HuggingFace language model endpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
chat_llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=hf_token,
    temperature=0.7,
    model_kwargs={"max_length": 128}
)

# Prompt template for RetrievalQA
prompt_str = """
Answer the question briefly on the basis of the given context.
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(prompt_str)

# Define retrieval setup
number_chunks = 3
retrieval = vectordb.as_retriever(search_kwargs={"k": number_chunks})

# Define the question handler and setup for LangChain chain
question_fetcher = itemgetter("question")
setup = {"question": question_fetcher, "context": question_fetcher | retrieval}
chain = (setup | prompt | chat_llm)

# Streamlit Input and Execution
st.write("## National AI policy GPT")

query = st.text_input("Enter your question:")

if st.button("Submit"):
    if query:
        response = chain.invoke({"question": query})
        st.write("### Response:")
        st.write(response)
    else:
        st.write("Please enter a question to get a response.")
