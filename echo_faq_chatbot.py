import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Local (free) embeddings
from langchain_huggingface import HuggingFaceEmbeddings  # preferred if available
# If you don't have langchain-huggingface installed, use this instead:
# from langchain_community.embeddings import HuggingFaceEmbeddings

# Local LLM via Ollama
from langchain_ollama import ChatOllama

from langchain_classic.chains import RetrievalQA


def main():
    # 1) Load FAQ text (must exist in same folder or provide full path)
    faq_path = "echo_faq.txt"
    if not os.path.exists(faq_path):
        raise FileNotFoundError(
            f"Missing {faq_path}. Create it in the same folder as this script."
        )

    loader = TextLoader(faq_path, encoding="utf-8")
    documents = loader.load()

    # 2) Chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # 3) Local embeddings (downloads model first run)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4) Vector store (in-memory; you can persist if you want)
    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 5) Local LLM (Ollama)
    llm = ChatOllama(model="llama3", temperature=0)

    # 6) RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )

    print("EchoHelp Bot (OFFLINE). Type 'exit' to quit.")
    while True:
        query = input("User: ").strip()
        if query.lower() == "exit":
            break
        result = qa_chain.invoke({"query": query})
        print("Bot:", result["result"])
        print()


if __name__ == "__main__":
    main()
