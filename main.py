from dotenv import load_dotenv

load_dotenv(".env")


import os

print(os.getenv("CHROMA_DB_PATH"))

from pptTools_v2.src.pipeline import Pipeline

executor = Pipeline("test_1",path=[r"D:\Asus\Srinivas\development\GenAI\code\open_park\DTC_Trends.pptx"])
executor.execute() # to execute Pipeline
