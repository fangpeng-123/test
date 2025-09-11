from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(model="text-embedding-v1")

query_vector = embeddings.embed_query("通义千问支持哪些向量模型？")
print("向量维度：", len(query_vector))

doc_vectors = embeddings.embed_documents([
    "通义千问是阿里云的大模型。",
    "DashScope 提供统一系列 API。"
])

print("文档向量维度：", len(doc_vectors), "x", len(doc_vectors[0]))