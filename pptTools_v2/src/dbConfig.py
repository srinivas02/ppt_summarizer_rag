import os
import sqlite3
import traceback
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker
from langchain.embeddings.openai import OpenAIEmbeddings


Base = declarative_base()

class Collection(Base):
    __tablename__ = 'collections'

    id = Column(Integer, primary_key=True, autoincrement=True)
    pptx_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    last_modified = Column(DateTime, nullable=False)
    file_size = Column(Integer, nullable=False)
    hash_id = Column(String, nullable=False)
    collection_name = Column(String, nullable=False)
    embedding_model_name = Column(String, nullable=False)

class dbHandler:
    def __init__(self, init_vector_store=True, collection_name = None, embedding_function=None):
        db_filename = os.environ.get('DB_FILENAME', "local_database.db")
        self.collection_name = collection_name
        self.embedding_model_name = None
        # Initialize SQL Engine
        engine = create_engine(f'sqlite:///{db_filename}')
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()
        # Initialize Vector Store
        self.vector_store = None
        if init_vector_store:
            self.__ignite_vector_store__(collection_name, embedding_function)

    def __ignite_vector_store__(self, collection_name, embedding_function):
        if embedding_function is None:
            embedding_function = OpenAIEmbeddings()
        if os.environ.get("VDB_HOST", None) == "MILVUS":
            from langchain.vectorstores.milvus import Milvus
            self.vector_store = Milvus(
                connection_args={
                                "uri": os.environ["MILVUS_URI"], 
                                "token": os.environ["MILVUS_TOKEN"]})
            self.vector_store.embedding_func = embedding_function
            self.vector_store.collection_name = collection_name
        else:
            from langchain.vectorstores.chroma import Chroma
            os.makedirs(os.environ["CHROMA_DB_PATH"], exist_ok=True)
            
            self.vector_store = Chroma(persist_directory=os.environ["CHROMA_DB_PATH"], 
                                       collection_name=collection_name, 
                                       embedding_function=embedding_function)
            # self.vector_store.heartbeat()
        return self.vector_store

    def add_content_vdb(self, content):
        self.vector_store.add_documents(content)

    def insert_data_in_sql(self, **kwargs):
        new_collection = Collection(
                                    pptx_name=kwargs['file_name'],
                                    file_path=kwargs['file_path'],
                                    file_size=kwargs['file_size'],
                                    last_modified = kwargs['modified_date'],
                                    hash_id = kwargs['hash_id'],
                                    collection_name=kwargs['collection_name'],
                                    embedding_model_name=kwargs['embedding_model_name']
                                )
        self.session.add(new_collection)
        self.session.commit()
        # return self.cur.lastrowid
    
    def fetch_files_data_from_sql(self, collection_name):
        files = self.session.query(Collection).filter(Collection.collection_name == collection_name).all()
        return [file for file in files]
    def collection_name_exists(self, collection_name):
        return self.session.query(Collection).filter(Collection.collection_name == collection_name).count() > 0
    
    def delete_collection(self, collection_name):
        try:
            self.session.query(Collection).filter(Collection.collection_name == collection_name).delete()
            self.session.commit()
            # ToDo: [Delete Collection from vector database]
            return True
        except:
            print(traceback.format_exc())
            return False
    
    def delete_file_in_sql(self, collection_name, file_path):
        try:
            self.session.query(Collection).filter(
                                                    (Collection.collection_name == collection_name) &
                                                    (Collection.pptx_name == file_path)
                                                ).delete()
            self.session.commit()
            return True
        except:
            print(traceback.format_exc())
            return False
    def __delete_store_vectordb__(self):
        pass

    def close_sql_connection(self):
        self.conn.close()
        self.cur = None
        self.conn = None
        print("DB connection closed")
    
    

       