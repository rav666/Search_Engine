import json

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
data_path = project_root/'data'/'movies.json'

def load_movies()->list[dict]:
    with open(data_path, mode='r') as f:
        data = json.load(f)
        return data['movies']