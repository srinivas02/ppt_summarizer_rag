from pptTools_v2.agents.coversational import ConversationalAgent
from pptTools_v2.agents.summary import SummaryAgent
from pptTools_v2.agents.qna import QnAAgent


__all__ = ["ConversationalAgent", "SummaryAgent", "QnAAgent"]

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", max_tokens=4096)
llm_chain = LLMChain(llm=llm, prompt=prompt)