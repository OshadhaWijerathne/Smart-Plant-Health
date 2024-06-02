import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from functools import lru_cache 

from dotenv import load_dotenv
import os

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


PROMPT_TEMPLATE = """
Question: {question}

"{context}"

Answer: You are a chat bot to help farmers about crop diseases and you should give a explained answer to the question based on the context provided.
"""
@lru_cache
def prepare_db(CHROMA_PATH):
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory= CHROMA_PATH,
                    embedding_function=embedding_function)
        return db


def generate_prompt(results, query_text):
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt

def ask_from_llm(query_text: str, vectordb):
    
    results = vectordb.similarity_search_with_relevance_scores(query_text, k=3)
    #print(results)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return                   

    prompt = generate_prompt(results, query_text)
    #print(prompt)

    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]

    #print(sources)
    print(response_text)
    return response_text

def chatbot_func(vectordb):
    st.markdown("# Chatbot 🤖")
    st.sidebar.markdown("# Chatbot 🤖")
    st.write("Chatbot is under construction")

    form_input = st.text_input('Enter Question')
    submit = st.button("Ask")

    if submit:
        #st.write(form_input)  Diseases of Citrus
        result = ask_from_llm(form_input, vectordb)
        st.write(result.content)