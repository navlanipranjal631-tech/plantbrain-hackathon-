import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("Available models that support embedContent:\n")
for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print(m.name)

print("\nAvailable models that support generateContent:\n")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)