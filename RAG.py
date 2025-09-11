from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


def load_docs(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return [Document(page_content=text)]

docs = load_docs("/media/yls/1T硬盘6/文档/test.txt")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
docsearch = FAISS.from_documents(texts, embeddings)

ollm = OllamaLLM(model="qwen3:8b")

from langchain.chains.question_answering import load_qa_chain
prompt_template = """请根据以下提供的上下文信息回答问题
{context}

问题：{question}
"""
QA_CHAIN_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain = load_qa_chain(llm=ollm, prompt=QA_CHAIN_PROMPT)

query = input("请输入你的问题：")
docs = docsearch.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)
print(f"答案：{answer}")