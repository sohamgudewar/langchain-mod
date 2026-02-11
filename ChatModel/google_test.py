import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# In 2026, 'gemini-2.0-flash' is the stable standard.
# We remove 'client_options' to avoid the Pydantic ValidationError.model = ChatGoogleGenerativeAI(

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", # Or "gemini-2.5-flash-lite" if available in your region
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

try:
    print("--- Connecting to Gemini 2.0 ---")
    result = model.invoke("5 lines write a trap music with playboi carti style(cuss words allowed)")
    print("\nSUCCESS!")
    print(result.content)
except Exception as e:
    print(f"\n--- ERROR ---\n{e}")
