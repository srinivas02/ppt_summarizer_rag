import os
from llama_index.vector_stores import MilvusVectorStore
from llama_index import ServiceContext, StorageContext, LLMPredictor
from llama_index.indices.vector_store import VectorStoreIndex
from langchain import OpenAI
from llama_index import ServiceContext, LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
import tiktoken

from langchain import hub
from langchain.prompts import PromptTemplate
from llama_index.prompts import LangchainPromptTemplate
from llama_index.schema import Document
import sqlite3
from datetime import datetime
from mavericks.src import *
from mavericks.src.extractor import PPTXExtractor
from mavericks.common.utils import qa_prompt, get_embedding_model


pptx_path = "/content/drive/MyDrive/ppts/DTC_Trends.pptx"
embeding_model_name = "text-embedding-ada-002"
top_k = 5
collection_name = "team1"
streaming = True

query = "what is title of fisrt slide?"


class MaverickRagAgent(PPTXExtractor):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.filepath = filepath
        self.documnets = [Document.from_langchain_format(doc) for doc in self.get_slide_content()]
        db_filename = 'maverick.db'
        self.conn = sqlite3.connect(db_filename)
        self.cur = self.conn.cursor()
        self.collection_name = self.embedding_model_name = None

    def check_documents_present(self):
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS vector_storage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pptx_path_name TEXT NOT NULL,
                inserted_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                modify_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                collection_name TEXT NOT NULL,
                embedding_model_name TEXT NOT NULL
            );
            '''
        self.cur.execute(create_table_sql)
        self.conn.commit()
        self.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        query = '''
        SELECT EXISTS(
            SELECT 1
            FROM vector_storage
            WHERE pptx_path_name = ?
            AND collection_name = ?
            AND embedding_model_name = ?
        );
        '''
        self.cur.execute(query, (self.filepath, self.collection_name, self.embedding_model_name))
        exists = self.cur.fetchone()[0]
        return exists == 1


    def ignite(self, embeding_model_name, top_k=5, collection_name="team_1", streaming=True):
        self.embeding_model_name = embeding_model_name
        self.top_k = top_k
        self.collection_name = collection_name
        self.streaming = streaming
        mv_store = MilvusVectorStore(uri=milvus_uri, token=token, dim=1536, collection_name=collection_name)

        lc_prompt_tmpl = LangchainPromptTemplate(
            template=qa_prompt,
            template_var_mappings={"query_str": "question", "context_str": "context"},
        )
        embed_model = get_embedding_model(embeding_model_name)
        text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=1024,
        chunk_overlap=20,
        backup_separators=["\n"],
        tokenizer=tiktoken.encoding_for_model(os.environ["OPENAI_MODEL_NAME"]).encode
        )
        prompt_helper = PromptHelper(
        context_window=4096, 
        num_output=256, 
        chunk_overlap_ratio=0.1, 
        chunk_size_limit=None
        )

        service_context = ServiceContext.from_defaults(
        llm=OpenAI(temperature=0, model=os.environ["OPENAI_MODEL_NAME"], max_tokens=2048),
        embed_model=embed_model,
        node_parser=text_splitter,
        prompt_helper=prompt_helper
        )
        storage_context = StorageContext.from_defaults(vector_store=mv_store)
        if self.check_documents_present():
            index = VectorStoreIndex.from_vector_store(
                mv_store, storage_context=storage_context, service_context=service_context
            )
        else:
            index = VectorStoreIndex.from_documents(
                self.documnets, storage_context=storage_context, service_context=service_context
            )
        self.query_engine = index.as_query_engine(service_context=service_context, similarity_top_k=top_k, streaming=streaming)
        self.query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
        )
    def query(self, query):
        streaming_response = self.query_engine.query(query)
        if self.streaming:
            streaming_response.print_response_stream()
        else:
            print(f"{streaming_response.response}")