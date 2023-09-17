import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


def getAnswer(question):
    if st.session_state.conversation:
        response = st.session_state.conversation({"question": question})
        st.session_state.chat_history = response["chat_history"]

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(
                    f'<div class="message-container"><div class="user-bubble"><div class="name">User</div><div>{message.content}</div></div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.write(
                    f'<div class="message-container"><div class="bot-bubble"><div class="name">Bot</div><div>{message.content}</div></div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.warning("Please click the 'Process' button to initialize the conversation.")


load_dotenv()
st.set_page_config(page_title="PDF Chatter", page_icon=":books:")
st.write(
    """
    <style>
    .message-container {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        margin-bottom: 10px;
    }

    .name {
        font-weight: bold;
        margin-bottom: 2px;
    }

    .user-bubble {
        background-color: #cfe2ff;
        border-radius: 10px;
        padding: 5px 10px;
        margin: 5px;
    }
    
    .bot-bubble {
        background-color: #e2ffc7;
        border-radius: 10px;
        padding: 5px 10px;
        margin: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

st.header("Chat with PDF")
question = st.text_input("Ask a question about your PDF")
if question:
    getAnswer(question)


def processPDF(pdf):
    text = ""
    reader = PdfReader(pdf)
    for page in reader.pages:
        text += page.extract_text()
    return text


def getTextChunks(text):
    textSplitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = textSplitter.split_text(text)
    return chunks


def getVectorStore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorStore = FAISS.from_texts(chunks, embeddings)
    return vectorStore


def getConversationChain(vectorStore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversationChain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorStore.as_retriever(), memory=memory
    )
    return conversationChain


with st.sidebar:
    st.subheader("Upload docs")
    pdf = st.file_uploader("Upload your PDF here")
    if st.button("Process"):
        if pdf:
            with st.spinner("Processing"):
                text = processPDF(pdf)
                textChunks = getTextChunks(text)
                vectorStore = getVectorStore(textChunks)
                st.session_state.conversation = getConversationChain(vectorStore)
                st.write("Processed, start chatting :)")
        else:
            st.warning("Please upload a PDF before clicking the 'Process' button.")
