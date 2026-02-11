from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

try:
    print("--- Connecting to huggingfacehub---")
    result = model.invoke("5 lines write a trap music with playboi carti style(cuss words allowed)")
    print(result.content)
except Exception as e:
    print(f"\n--- ERROR ---\n{e}")
