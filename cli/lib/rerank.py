import json
import os

from dotenv import load_dotenv

from lib.search_utils import PROMPT_PATH

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")
print(f"Using key {api_key[:6]}...")

from google import genai

model = "gemini-2.5-flash"
client = genai.Client(api_key=api_key)


def individual_rerank(query, documents):
    with open(PROMPT_PATH / 'individual_rerank.md') as f:
        prompt = f.read()
    results = []
    for doc in documents:
        prompt1 = prompt.format(query=query, title=doc["title"], description=doc["description"])
        response = client.models.generate_content(model=model, contents=prompt1)
        clean_response_text = (response.text or "0").strip()
        data = {**doc, 'rerank_response': int(clean_response_text)}

        print(data["title"], data["rerank_response"], sep='\t')

        results.append(data)

        # doc["rerank_response"] =
    results = sorted(results, key=lambda r: r['rerank_response'], reverse=True)
    return results


def batch_rerank(query, documents):
    with open(PROMPT_PATH / 'batch_rerank.md') as f:
        prompt = f.read()
    mtemp = '''<movie id={idx}=>{title}>\n{desc}\n</movie>\n'''
    doc_list_str = ''
    for idx, doc in enumerate(documents):
        doc_list_str += mtemp.format(idx=idx, title=doc["title"], desc=doc["description"])

    prompt1 = prompt.format(
        query=query,
        doc_list_str=doc_list_str)
    response = client.models.generate_content(model=model, contents=prompt1)

    response_parsed = json.loads(response.text.strip('```json').strip('```').strip())
    results = []
    for idx, doc in enumerate(documents):
        results.append({**doc, 'rerank_score': response_parsed.index(idx)})
    results = sorted(results, key=lambda r: r['rerank_score'], reverse=False)
    return results
