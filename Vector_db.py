from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
import os

os.environ["DASHSCOPE_API_KEY"]="sk-b594b133f3274e368d577ea68e09e256"

loader = TextLoader("/media/yls/1T硬盘7/文档/test.txt", encoding="utf-8")
raw__docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
split_docs = text_splitter.split_documents(raw__docs)

embedding = DashScopeEmbeddings(model="text-embedding-v1")
db = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory="/media/yls/1T硬盘7/文档/Chroma_db"
)