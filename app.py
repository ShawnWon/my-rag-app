import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

st.title("Shawn's RAG Chatbot")

embeddings = OllamaEmbeddings(
    model="llama3"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

llm = ChatOllama(model="llama3")

query = st.text_input("Ask a question")

if query:
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    st.write(response.content)