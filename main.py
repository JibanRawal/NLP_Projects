# Importing the necessary files and model
from flask import Flask  # Web platform
from langchain_community.llms import Ollama  # Local open source model
from langchain_core.prompts import ChatPromptTemplate  # Prompt template
from langchain_core.output_parsers import StrOutputParser  # Chat message to string

from langchain_community.document_loaders import WebBaseLoader   # Web scrapping
from langchain_community.embeddings import OllamaEmbeddings  # Embedding model

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

from langchain_core.messages import HumanMessage, AIMessage



app = Flask(__name__)

# Initialize the llama2 model
llm = Ollama(model="llama2")
# Initialize the mistral model
# llm=Ollama(model="mistral")

@app.route('/')
def index():
    # Prompt templates convert raw user input to better input to LLM
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()  # Convert the chat message to the string
    chain = prompt | llm | output_parser # Chanining the lLM chain
    result=chain.invoke({"input": "how can langsmith help with testing?"}) # Invoking the query
    return result # returning the result


@app.route('/Retieval_llm')
def scrapping():
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()
    embeddings = OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": "how can langsmith help with testing?"})
    print(response["answer"])

@app.route('/Conversational_llm')
def conversation():
    # First we need a prompt that we can pass into an LLM to generate this search query
    loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
    docs = loader.load()
    embeddings = OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation")
      ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
    chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "Tell me how"
    })

    print(response["answer"])


conversation()


if __name__ == '__main__':
    app.run(debug=True)
