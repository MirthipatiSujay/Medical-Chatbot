from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import traceback

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Lazy-loaded resources (initialized on first request)
rag_chain = None

def get_rag_chain():
    global rag_chain
    if rag_chain is None:
        print("Initializing ML models and connections...")
        embeddings = download_embeddings()

        index_name = "medical-chatbot"
        docSearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings,
        )
        retriever = docSearch.as_retriever(search_type="similarity", search_kwargs={"k": 8})

        chatModel = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("Initialization complete!")
    return rag_chain

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/health")
def health():
    status = {
        "pinecone_key_set": bool(PINECONE_API_KEY),
        "google_key_set": bool(GOOGLE_API_KEY),
        "rag_chain_loaded": rag_chain is not None,
    }
    return jsonify(status)

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print(f"User message: {msg}")
        chain = get_rag_chain()
        response = chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"ERROR in /get: {error_msg}")
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port = 8080)
