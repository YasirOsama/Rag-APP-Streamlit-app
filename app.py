import os
import streamlit as st
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from operator import itemgetter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

# Initialize tools and resources
st.title("National AI Policy and Wikipedia/Arxiv Query Tool")

# Wikipedia setup
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# Arxiv setup
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# LangSmith documentation loader and vector store
loader = WebBaseLoader('https://docs.smith.langchain.com/')
docs = loader.load()

embed_model = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
vectordb = FAISS.from_documents(documents, embed_model)
retriever = vectordb.as_retriever()

# Create retriever tool for LangSmith
retriever_tool = create_retriever_tool(retriever, "langsmith_search",
                                       "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!")

# Combine all tools into a list
tools = [wiki, arxiv, retriever_tool]

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
st.write("## Query Wikipedia, Arxiv, and LangSmith Documentation")

query = st.text_input("Enter your question:")

if st.button("Submit"):
    if query:
        response = chain.invoke({"question": query})
        st.write("### Response:")
        st.write(response)
    else:
        st.write("Please enter a question to get a response.")
