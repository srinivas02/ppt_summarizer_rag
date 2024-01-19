from langchain import hub

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


summarize_prompt_templates = """Create a comprehensive summary encompassing the information present in all documents. Don't Provide any false information in Summary. Organize your response in detailed paragraph to capture the key details, employing tables and chart data where applicable to enhance clarity. Not needed to provide sumary slide wise but make sure you summary provides a landscape view of all slides. Ensure that your summary is detailed and highlights crucial information. Use the designated triple backquotes to encapsulate the text. Please provide output in markdown format.
```{text}```
BULLET POINT SUMMARY:
    """

chat_context_prompt = """You are a helpful, respectful and honest assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Include slide references where the information is found. Provide the sources at the end where you found the information with slide source and slide number. Please follow the following example in <EXAMPLE></EAXMPLE> block for you reference:  

<EXAMPLE>
Query: 
```What is Full Form of DTC?```
Response: 
```The full form of the DTC is "Direct To Consumer" [1].
1: Slide #1 :  - /data/abc.pptx```
</EXAMPLE>
Here are the relevant documents for the context:
Context: {context_str} 
Instruction: Use the previous chat history, or the context above, to interact and help the user.
Please provide output in markdown format."""
qa_prompt = hub.pull("rlm/rag-prompt")

qa_edited_prompt = """You are a helpful, respectful and honest assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Include slide references where the information is found. Provide the sources at the end where you found the information with slide source and slide number. Please follow the following example in <EXAMPLE></EAXMPLE> block for you reference:  

<EXAMPLE>
Question: 
```What is Full Form of DTC?```
Answer: 
```The full form of the DTC is "Direct To Consumer" [1].
1: Slide #1 :  - /data/abc.pptx```
</EXAMPLE>
Please provide output in markdown format.

Question: {question} 
Context: {context} 
Answer:"""
qa_prompt.messages[0] = qa_prompt.messages[0].from_template(qa_edited_prompt)