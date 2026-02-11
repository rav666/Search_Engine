import json
import re
from collections import defaultdict

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
            results.append({'score': sc, 'title': doc['title'], 'description': doc['description'][100], })
        return results


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self):
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.metadata_path = CACHE_PATH / 'chunk_metadata.json'
        self.embeddings_path = CACHE_PATH / 'chunk_embeddings.npy'

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        all_chunks = []
        chunk_metadata = []
        for midx, doc in enumerate(documents):
            if doc['description'].strip() == '':
                continue
            chunks = semantic_chunk(doc['description'], overlap=1, max_chunk_size=4)
            all_chunks += chunks
            for cidx in range(len(chunks)):
                chunk_metadata.append({"movie_idx": midx + 1, "chunk_idx": cidx + 1, "total_chunks": len(chunks)})
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        np.save(self.embeddings_path, self.chunk_embeddings)
        with open(self.metadata_path, 'w') as f:
            json.dump({'chunks': chunk_metadata, 'total_chunks': len(all_chunks)}, f, indent=4)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        if self.embeddings_path.exists() and self.metadata_path.exists():
            self.chunk_embeddings = np.load(self.embeddings_path)
            with open(self.metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        query_emb = self.generate_embeddings(query)
        chunk_scores = []
        movie_scores = defaultdict(lambda: 0)
        for idx in range(len(self.chunk_embeddings)):
            chunk_embedding = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata['chunks'][idx]
            midx, cidx = metadata['movie_idx'], metadata['chunk_idx']
            sim = cosine_similarity(query_emb, chunk_embedding)
            chunk_scores.append({
                'movie_idx': midx,
                'chunk_idx': cidx,
                'score': sim,
            })
            movie_scores[midx] = max(movie_scores[midx], sim)
        movie_scores_sorted = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        res = []
        for midx, score in movie_scores_sorted[:limit]:
            doc = self.document_map[midx]
            res.append(
                {
                    "id": doc['id'],
                    "title": doc['title'],
                    "document": doc['description'][:100],
                    'score': round(score, 4),
                    'metadata': {}

                }
            )
        return res


def searched_chunks(query, limit=10):
    ss = ChunkedSemanticSearch()
    movies = load_movies()
    embeddings = ss.load_or_create_chunk_embeddings(movies)
    result = ss.search_chunks(query, limit)
    for i, res in enumerate(result):
        print(f"\n{i + 1}. {res['title']} (score: {res['score']:.4f})")
        print(f"{res['document']}...")



def embed_chunks():
    movies = load_movies()
    css = ChunkedSemanticSearch()
    embeddings = css.load_or_create_chunk_embeddings(movies)
    print(f"generated{len(embeddings)} chunked embeddings")

def semantic_chunk(text, overlap=0, max_chunk_size=4):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # print(f"sentences: {sentences}")
    chunks = []
    step_size = max_chunk_size - overlap
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i:i + max_chunk_size]
        if len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
    # print(f"chunks: {chunks}, max_chunk_size: {max_chunk_size}, overlap: {overlap}")
    return chunks


def chunk_text_semantic(text, overlap=0, max_chunk_size=4):
    chunks = semantic_chunk(text, overlap, max_chunk_size)
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


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
