from dotenv import load_dotenv

load_dotenv()
import os
print(os.environ["OPENAI_API_KEY"])