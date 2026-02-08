from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embeddings(self, text):
        if not text or not text.strip():
            raise ValueError('text is empty')
        return self.model.encode([text])[0]


def verify_model():
    ss = SemanticSearch()
    print(f"Model Loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embeddings = ss.generate_embeddings(text)
    print(f"embedding text: {text}")
    print(f"first three dimensions: {embeddings[:3]}")
    print(f"dimensions: {embeddings.shape[0]}")
