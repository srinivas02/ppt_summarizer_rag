import os
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv(".env")

embedding_model_host = os.environ.get("EMBEDDING_MODEL_HOST", "OPENAI")
embedding_model = None
if embedding_model_host == "OPENAI":
    embedding_model =  OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL_NAME"))
        # embed_batch_size=embed_batch_size)
else:
    embedding_model =  HuggingFaceEmbedding(
        model_name=model_name,
        embed_batch_size=embed_batch_size)

vdb_host = os.environ.get("VDB_HOST", "MILVUS")
vector_store = None
if vdb_host == "MILVUS":
    from vectorstores.vectorstore import MilvusVectorStore
    vector_store = MilvusVectorStore(uri=os.environ["MILVUS_URI"], token=os.environ["MILVUS_TOKEN"], dim=1536, collection_name=collection_name)
else:
    import chromadb
    os.makedirs(os.environ["CHROMA_DB_PATH"], exist_ok=True)
    vector_store = chromadb.PersistentClient(path=os.environ["CHROMA_DB_PATH"])
    vector_store.heartbeat()