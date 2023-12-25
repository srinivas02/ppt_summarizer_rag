import json
import os
# from img2table.ocr import TesseractOCR
# from img2table.document import Image as img
from langchain.text_splitter import CharacterTextSplitter
from langchain import hub
from langchain.prompts import PromptTemplate

import numpy as np
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding

chartTypeGlossary = json.load(open(os.path.join(os.getcwd(),'mavericks/config/glossary.json')))

# Map
map_prompt_template = """The following is a set of documents
{docs}
Use the document list provided to create thorough summaries for each document. Incorporate tables and charts if they are available in the slides. Avoid adding information not present in the documents or creating false summaries.
Summary:
"""
"""Utilize this document list to provide detailed summaries for each document, incorporating tables and charts wherever it applicable to reinforce your analysis. Don't try to make up the text by yourself. Don't provide false information or false summarization.
Helpful Answer:"""


# Reduce
reduce_prompt_template = """Below is collection of summaries:
{docs}
Consolidate these materials into a concise summary organized by slides. Utilize tables and chart data if it presents in data to reinforce your summary. Please provide output in markdown format. Please don't share false information or false summarization.
Summary:"""

summarize_prompt_templates = """Create a comprehensive summary encompassing the information present in all documents. Organize your response in detailed paragraph to capture the key details, employing tables and chart data where applicable to enhance clarity. Not needed to provide sumary slide wise but make sure you summary provides a landscape view of all slides. Ensure that your summary is detailed and highlights crucial information. Use the designated triple backquotes to encapsulate the text. Please provide output in markdown format.
```{text}```
BULLET POINT SUMMARY:
    """

qa_prompt = hub.pull("rlm/rag-prompt")
my_prompt = """You are a helpful, respectful and honest assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Include slide references where the information is found. Provide the sources at the end where you found the information with slide source and slide number. Please follow the following example in <EXAMPLE></EAXMPLE> block for you reference:  

<EXAMPLE>
The full form of the DTC is "Direct To Consumer" [1].

1: Slide # (Slide Number) :  - /data/abc.pptx
</EXAMPLE>
Please provide output in markdown format.

Question: {question} 
Context: {context} 
Answer:"""

qa_prompt.messages[0] = qa_prompt.messages[0].from_template(my_prompt)
chat_context_prompt = """You are a helpful, respectful and honest assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Include slide references where the information is found. Provide the sources at the end where you found the information with slide source and slide number. Please follow the following example in <EXAMPLE></EAXMPLE> block for you reference:  

<EXAMPLE>
The full form of the DTC is "Direct To Consumer" [1].

1: Slide # (Slide Number) :  - /data/abc.pptx
</EXAMPLE>
Here are the relevant documents for the context:
Context: {context_str} 
Instruction: Use the previous chat history, or the context above, to interact and help the user.
Please provide output in markdown format."""

def get_embedding_model(model_name, embed_batch_size=100):
    if model_name == "text-embedding-ada-002":
            return OpenAIEmbedding(
                model=model_name,
                embed_batch_size=embed_batch_size,
                api_key=os.environ["OPENAI_API_KEY"])
    else:
        return HuggingFaceEmbedding(
            model_name=model_name,
            embed_batch_size=embed_batch_size)

def get_splitted_docs(docs, chunk_size=4024, chubk_overlap=0):
    """
    Splits a given document into chunks using a character-based text splitter.

    Args:
        docs (str): The document to be split into chunks.

    Returns:
        list: A list of chunks generated from the document.
    """
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chubk_overlap
    )
    return text_splitter.split_documents(docs)



# def extract_image_content(shape):
#         """
#         [NOT ADDED in MODULE]
#         Extracts table data from the given image shape using OCR.

#         Parameters:
#         - shape: Image shape object.

#         Returns:
#         - list: List containing DataFrames with image content data.
#         """
#         image_content_data = []
#         try:
#             image_stream = io.BytesIO(shape.image.blob)
#             image = Image.open(image_stream)
#             image.convert('L').save("temp.jpg")

#             tab_image = img(src="temp.jpg")
#             data = tab_image.extract_tables(ocr=TesseractOCR(
#                 lang="eng"), borderless_tables=True, min_confidence=0)
#             if len(data) == 0:
#                 data = tab_image.extract_tables(
#                     ocr=TesseractOCR(lang="eng"), min_confidence=0)
#             if len(data) > 0:
#                 data = pd.concat([i.df for i in data], axis=1)
#             else:
#                 data = None
#             os.remove("temp.jpg")
#         except Exception as e:
#             data = None
#             print(traceback.format_exc())
#         image_content_data.append(data)
#         return image_content_data