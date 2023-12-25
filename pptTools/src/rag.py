import os
import sqlite3
from datetime import datetime
from mavericks.src import *
from mavericks.src.extractor import PPTXExtractor
from mavericks.common.utils import qa_prompt, get_embedding_model, chat_context_prompt
from llama_index.vector_stores import MilvusVectorStore
from llama_index import ServiceContext, StorageContext, LLMPredictor
from llama_index.indices.vector_store import VectorStoreIndex
from langchain import OpenAI
from llama_index import LLMPredictor, OpenAIEmbedding, PromptHelper
from llama_index.llms import OpenAI
from llama_index.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
import tiktoken
from langchain import hub
from langchain.prompts import PromptTemplate
from llama_index.prompts import LangchainPromptTemplate
from llama_index.schema import Document
from llama_index.memory import ChatMemoryBuffer

class MaverickRagAgent(PPTXExtractor):
    """MaverickRagAgent extracts information from PPTX files and provides querying capabilities.

    Args:
        filepath (str): The path to the PPTX file.

    Attributes:
        filepath (str): The path to the PPTX file.
        documents (list): List of documents extracted from the PPTX file.
        conn: SQLite database connection.
        cur: SQLite database cursor.
        collection_name (str): Name of the collection in the vector storage.
        embedding_model_name (str): Name of the embedding model used.
    """

    def __init__(self, filepath):
        super().__init__(filepath)
        self.filepath = filepath
        self.documents = [Document.from_langchain_format(doc) for doc in self.get_slide_content()]
        db_filename = 'maverick.db'
        self.conn = sqlite3.connect(db_filename)
        self.cur = self.conn.cursor()
        self.collection_name = self.embedding_model_name = None

    def check_documents_present(self):
        """Check if documents are present in the vector storage.

        Returns:
            bool: True if documents are present, False otherwise.
        """
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

    def ignite(self, embedding_model_name, top_k=5, collection_name="team1", streaming=True, chat_mode = False):
        """Initialize the MaverickRagAgent with the specified parameters.

        Args:
            embedding_model_name (str): Name of the embedding model.
            top_k (int, optional): Top k results to retrieve. Defaults to 5.
            collection_name (str, optional): Name of the vector storage collection. Defaults to "team_1".
            streaming (bool, optional): Enable streaming response. Defaults to True.
        """
        self.embedding_model_name = embedding_model_name
        self.top_k = top_k
        self.collection_name = collection_name
        self.streaming = streaming
        self.chat_mode = chat_mode
        mv_store = MilvusVectorStore(uri=os.environ["MILVUS_URI"], token=os.environ["MILVUS_TOKEN"], dim=1536, collection_name=collection_name)

        lc_prompt_tmpl = LangchainPromptTemplate(
            template=qa_prompt,
            template_var_mappings={"query_str": "question", "context_str": "context"},
        )
        embed_model = get_embedding_model(embedding_model_name)
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
                self.documents, storage_context=storage_context, service_context=service_context
            )
        
        if self.chat_mode:
            memory = ChatMemoryBuffer.from_defaults(token_limit=3900)
            self.query_engine = index.as_chat_engine(service_context=service_context, 
                                                     chat_mode="condense_plus_context", 
                                                     memory = memory, 
                                                     verbose = False, 
                                                     similarity_top_k=top_k, 
                                                     streaming=streaming, 
                                                     context_prompt = chat_context_prompt)
        else:
            self.query_engine = index.as_query_engine(service_context=service_context, similarity_top_k=top_k, streaming=streaming)
            self.query_engine.update_prompts(
                {"response_synthesizer:text_qa_template": lc_prompt_tmpl}
            )


    def chat(self, query):
        """Perform a query using the initialized query engine and display the results.

        Args:
            query (str): The query string.
        """
        # query = {}
        # while True:
        if self.chat_mode:
            self.response = self.query_engine.chat(f'Use the tool to answer: {query}')
        else:
            self.response = self.query_engine.query(query)
        if self.streaming:
            print(f"{self.response.response}")
            # response.print_response_stream()
        else:
            print(f"{self.response.response}")
