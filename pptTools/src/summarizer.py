from langchain.chains import (
    MapReduceChain, MapReduceDocumentsChain, ReduceDocumentsChain)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.tools import Tool
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import initialize_agent, AgentType

from pptTools.common.utils import get_splitted_docs, map_prompt_template, reduce_prompt_template, summarize_prompt_templates
from pptTools.src.extractor import PPTXExtractor

from requests import ConnectionError
from openai import APIConnectionError
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    

class SummarizerAgent(PPTXExtractor):
    """
    A class for summarizing content extracted from PowerPoint presentations.

    Attributes:
    - pptx_path (str): The file path of the PowerPoint presentation.
    - model_name (str): The name of the language model to be used (default is os.environ["OPENAI_MODEL_NAME"]).
    - temperature (int): The temperature parameter for controlling randomness in model responses (default is 0).
    """

    def __init__(self, pptx_path, model_name=os.environ["OPENAI_MODEL_NAME"], temperature=0):
        """
        Initializes the SummarizerAgent with the specified parameters.

        Parameters:
        - pptx_path (str): The file path of the PowerPoint presentation.
        - model_name (str): The name of the language model to be used (default is os.environ["OPENAI_MODEL_NAME"]).
        - temperature (int): The temperature parameter for controlling randomness in model responses (default is 0).
        """
        super().__init__(pptx_path)
        self.model_name = model_name
        self.temperature = temperature

    def extractor(self):
        """
        Extracts the content from the PowerPoint presentation and prints the raw documents.
        """
        self.raw_docs = self.get_slide_content()

    def preprocessor(self):
        """
        Prepares the extracted raw documents by splitting them into smaller units.
        """
        self.docs = get_splitted_docs(self.raw_docs)

    def detailed_summary(self):
        """
        Generates a detailed summary of the documents using language models.

        Returns:
        - str: The detailed summary.
        """
        logging.info("Starting detailed summarizing, Hold on for sometime....")
        docs = self.docs

        # Define prompt
        prompt_template = summarize_prompt_templates
        prompt = PromptTemplate.from_template(prompt_template)

        # Define LLM chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", max_tokens=4096)
        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # Define StuffDocumentsChain
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        logging.info("Detailed Summarization Completed!!!")

        return "Detailed Summary:\n" + stuff_chain.run(docs)

    def slide_summarizer(self) -> str:
        """
        Generates concise summaries for each slide and consolidates them into a concise summary organized by slides.

        Returns:
        - str: The concise summary.
        """
        try:
            logging.info("Starting summarizing, Hold on for sometime....")
            llm = ChatOpenAI(temperature=0, model_name=os.environ["OPENAI_MODEL_NAME"])

            # Map
            map_template = map_prompt_template

            map_prompt = PromptTemplate(template=map_template, input_variables=["docs"])
            map_chain = LLMChain(llm=llm, prompt=map_prompt)

            # Reduce
            reduce_template = reduce_prompt_template
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


    def init(self):
        """
        Initiates the Summarizer, performs extraction, preprocessing, and logs completion.
        """
        logging.info("PPT Summarizer in action....")
        self.extractor()
        logging.info("Extraction Completed!!")
        self.preprocessor()
        logging.info("Preprocessing Completed!!")

