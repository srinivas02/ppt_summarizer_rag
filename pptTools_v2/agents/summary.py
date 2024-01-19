from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from pptTools_v2.templates.prompts import summarize_prompt_templates
from . import llm, llm_chain
class SummaryAgent:
    def __init__(self, vector_connection):
        self.vector_connection = vector_connection
        self.documents = self.vector_connection.get()["documents"]
    
    def run(self):
        print(self.documents)
    
    
    def slide_summarizer(self) -> str:
        """
        Generates concise summaries for each slide and consolidates them into a concise summary organized by slides.

        Returns:
        - str: The concise summary.
        """
        try:
            # Map
            map_prompt = PromptTemplate(template=map_template, input_variables=["docs"])

            # Reduce
            reduce_prompt = PromptTemplate(template=reduce_template, input_variables=["docs"])

            # Run chain
            reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
            combine_documents_chain = StuffDocumentsChain(
                llm_chain=reduce_chain, document_variable_name="docs"
            )
            reduce_documents_chain = ReduceDocumentsChain(
                combine_documents_chain=combine_documents_chain,
                collapse_documents_chain=combine_documents_chain,
                token_max=16000,
            )
            map_reduce_chain = MapReduceDocumentsChain(
                llm_chain=map_chain,
                reduce_documents_chain=reduce_documents_chain,
                document_variable_name="docs",
                return_intermediate_steps=False
            )
            output = "Concise summary across all slides:\n" + map_reduce_chain.run(self.raw_docs)
            logging.info("Slide-wise Summarization Completed!!!")
            return output
        except APIConnectionError:
            logging.error("APIConnectionError: Please try later")
            return None