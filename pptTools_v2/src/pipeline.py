from pptTools_v2.src.contentInfuser import readerFiles
from pptTools_v2.agents import *

class Pipeline(readerFiles):
    def __init__(self, collection_name: str, agent_type: str = "chat", path = None):
        super().__init__(collection_name, path)
        if agent_type == "summarizer":
            self.agent = SummaryAgent(self.vector_store)
        elif agent_type == "qna":
            self.agent = QnAAgent(self.vector_store)
        elif agent_type == "chat":
            self.agent = ConversationalAgent(self.vector_store)
        else:
            raise ValueError("Agent type not supported")
    
    def execute(self):
        self.agent.run()