import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import tempfile
import os

st.set_page_config(page_title="🇰🇿 Constitution of the Republic of Kazakhstan - AI Assistant")

st.title("🇰🇿 Constitution of the Republic of Kazakhstan - AI Assistant")
st.markdown("Download the Constitution .txt file:")

uploaded_file = st.file_uploader("Select file", type=["txt"])

# Загрузка и индексация файла
if uploaded_file:
    st.success(f"File {uploaded_file.name} successfully uploaded as TXT!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding="utf-8") as tmp_file:
        content = uploaded_file.read().decode("utf-8")
        tmp_file.write(content)
        tmp_filepath = tmp_file.name

    loader = TextLoader(tmp_filepath, encoding="utf-8")
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.from_documents(texts, embeddings)

    st.session_state.vectorstore = vectorstore
    os.remove(tmp_filepath)  # удалить временный файл

# Поле для ввода вопроса
question = st.text_input("Enter a question about the Constitution:")

if question:
    if "vectorstore" not in st.session_state:
        st.warning("First, download the Constitution file.")
    else:
        llm = Ollama(model="llama3")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(),
            return_source_documents=True
        )

        with st.spinner("Generating a response..."):
            result = qa_chain.invoke({"query": question})

        st.markdown("### Answer:")
        st.write(result["result"])

        with st.expander("Sources:"):
            for doc in result["source_documents"]:
                st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                st.markdown(doc.page_content[:500] + "...")
