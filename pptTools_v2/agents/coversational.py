from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from pptTools_v2.templates.prompts import summarize_prompt_templates

class ConversationalAgent:
    def __init__(self, vector_connection):
        self.vector_connection = vector_connection
        self.documents = self.vector_connection.get()
    
    def run(self):
        print(self.documents)
        