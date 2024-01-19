import os
import hashlib
from pptTools_v2.src.dbConfig import dbHandler
from pptTools_v2.src.pptExtractor import PPTXExtractor
from datetime import datetime
class readerFiles(PPTXExtractor, dbHandler):
    def __init__(self, collection_name, path=None):
        self.path = path
        self.collection_name = collection_name
        print("Collection Name::::::::::::", self.collection_name)
        self.valid_files = []
        PPTXExtractor.__init__(self)
        dbHandler.__init__(self, collection_name=self.collection_name)
        raw_files_metadata = [] 
        if self.path is not None:
            if self.collection_name_exists(self.collection_name):
                self.delete_collection(self.collection_name)
            raw_files_metadata = self.get_files_and_details()
            # ToDo: Compare existing files with raw files data
        else:
            assert self.collection_name_exists(self.collection_name), "Collection name does not exists... Provide files for Retreival"
        if len(raw_files_metadata) > 0:
            self.infuse_content_db(raw_files_metadata)

    def get_files_and_details(self):
        if isinstance(self.path, list):
            valid_files =  self.return_valid_files(self.path)
        else:
            valid_files = self.return_valid_files(os.listdir(self.path))
        file_metadata = []
        for file in valid_files:
            file_metadata.append(self.get_metadata(file))
        return file_metadata
    
    def get_metadata(self, file):
        metadata = {
            'file_name': os.path.basename(file),
            'file_path': os.path.dirname(file),
            'modified_date': datetime.fromtimestamp(os.path.getmtime(file)),
            'file_size': os.path.getsize(file),
            'hash_id': hashlib.md5(open(file,'rb').read()).hexdigest()
        }
        return metadata

    def return_valid_files(self, files: list):
        files_inclusion_criteria = os.getenv('FILES_INCLUSION_CRITERIA').split(',')
        return [file for file in files if file.endswith(tuple(files_inclusion_criteria))]
    
    def infuse_content_db(self, raw_files_metadata):
        for each_file in raw_files_metadata:
            self.set_filepath(os.path.join(each_file["file_path"], each_file["file_name"]))
            content = self.get_content()
            self.add_content_vdb(content)
            print(self.collection_name)
            each_file["collection_name"] = self.collection_name
            each_file["embedding_model_name"] = "openai"
            self.insert_data_in_sql(**each_file)
        pass

    def __set_content_vdb__(self, content):
        last_file_modified_date = self.get_param_value_from_sql("last_modified", self.filepath)
        if last_file_modified_date != self.metadata['last_modified']:
            self.content_to_vdb(content)
            self.update_collection_name(self.filepath)
        collection_name = self.get_param_value_from_sql("collection_name", self.filepath)
            # delete collection from vector store

    def __retrieve_collection_details__(self, collection_name):
        pass
    def __set_metadata_sql__(self, metadata):
        pass

        