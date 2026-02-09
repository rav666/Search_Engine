import numpy as np
from sentence_transformers import SentenceTransformer

from lib.search_utils import load_movies, CACHE_PATH


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = CACHE_PATH / 'embeddings.npy'

    def build_embeddings(self, documents):
        self.documents = documents
        movie_strings = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        embeddings = documents
        self.document_map = {}
        self.documents = documents
        # docs and embeddings match check
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.documents) == len(self.embeddings):
                return self.embeddings
        return self.build_embeddings(documents)

    def generate_embeddings(self, text):
        if not text or not text.strip():
            raise ValueError('text is empty')
        return self.model.encode([text])[0]

    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError('embeddings is None')
        query_embedding = self.generate_embeddings(query)
        similarities = []
        for doc_emb, doc in zip(self.embeddings, self.documents):
            _similarity = cosine_similarity(query_embedding, doc_emb)
            similarities.append((_similarity, doc))
        similarities.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sc, doc in similarities[:limit]:
            results.append({
                'score': sc,
                'title': doc['title'],
                'description': doc['description'][100],
            })
        return results


def fixed_sized_chunks(text, overlap: int, chunk_size=200):
    words = text.split()
    chunks = []
    step_size = chunk_size - overlap
    for i in range(0, len(words), step_size):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def chunk_text(text, overlap: int, chunk_size=200):
    chunks = fixed_sized_chunks(text, overlap, chunk_size)
    print(f"chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}.) {chunk}")


def search(query, limit=5):
    ss = SemanticSearch()
    movies = load_movies()
    ss.load_or_create_embeddings(movies)
    result = ss.search(query, limit)

    for idx, r in enumerate(result):
        print(f'{idx + 1}. score = {r['score']:.2f},{r["title"]}: {r["description"]}')


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)



def verify_model():
    ss = SemanticSearch()
    print(f"Model Loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"number of documents: {len(documents)}")
    print(f"embedding shapes: {embeddings.shape[0]} vectors in {embeddings.shape[1]} ")


def embed_text(text):
    ss = SemanticSearch()
    embeddings = ss.generate_embeddings(text)
    print(f"embedding text: {text}")
    print(f"first three dimensions: {embeddings[:3]}")
    print(f"dimensions: {embeddings.shape[0]}")


def embed_query_text(query):
    ss = SemanticSearch()
    embeddings = ss.generate_embeddings(query)
    print(f"embedding text: {query}")
    print(f"first three dimensions: {embeddings[:3]}")
    print(f"dimensions: {embeddings.shape[0]}")
