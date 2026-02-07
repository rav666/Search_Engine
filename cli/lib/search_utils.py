import json

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
data_path = project_root/'data'/'movies.json'
stopwords_path = project_root/'data'/'stopwords.txt'

def load_movies()->list[dict]:
    with open(data_path, mode='r') as f:
        data = json.load(f)
        return data['movies']
def load_stopwords():
    with open(stopwords_path, mode='r') as f:
        data = f.read().splitlines(keepends= False)
    return data