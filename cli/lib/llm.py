import os

from dotenv import load_dotenv

from lib.search_utils import PROMPT_PATH

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
print(f"Using key {api_key[:6]}...")

from google import genai

model = "gemini-2.5-flash"
client = genai.Client(api_key=api_key)


def generate_content(prompt, query):
    prompt = prompt.format(query=query)
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def correct_spellings(query):
    with open(PROMPT_PATH / 'spelling.md', mode='r') as f:
        prompt = f.read()
    return generate_content(prompt, query)
