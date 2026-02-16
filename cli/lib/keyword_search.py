import math
import os
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from lib.search_utils import load_movies, load_stopwords, CACHE_PATH, BM25_K1, BM25_B

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'
        self.term_frequencies = defaultdict(Counter)
        self.term_frequencies_path = CACHE_PATH / 'term_frequencies.pkl'
        self.doc_lengths = {}
        self.doc_lengths_path = CACHE_PATH / 'doc_lengths.pkl'


    def _add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def _get_avg_doc_length(self):
        import statistics
        return statistics.mean(self.doc_lengths.values())

    def get_document(self, term):
        result = sorted(self.index[term])
        return list(result)

    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("can only have 1 tokens")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

    def bm25_get_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("can only have 1 tokens")
        token = tokens[0]
        tf = self.term_frequencies[doc_id][token]
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self._get_avg_doc_length())
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def get_idf(self, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("can only have 1 tokens")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])

        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_tfidf(self, term, doc_id):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term: str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("can only have 1 tokens")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def bm25(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.bm25_get_tf(doc_id, term, k1=k1, b=b)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query, limit=5):
        query_tokens = tokenize_text(query)
        scores = {}
        for doc_id in self.docmap:
            score = 0
            for token in query_tokens:
                score += self.bm25(doc_id, token, BM25_K1, BM25_B)
            scores[doc_id] = score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = sorted_scores[:limit]
        formatted_results = []
        for doc_id, score in results:
            title = self.docmap[doc_id]['title']
            formatted_results.append({
                "doc_id": doc_id,
                "title": title,
                "score": score,
                "description": self.docmap[doc_id]['description'],
            })
        return formatted_results





    def build(self):
        movies = load_movies()
        for movie in movies:
            text = f"{movie['title']}, {movie['description']}"
            self._add_document(movie['id'], text)
            self.docmap[movie['id']] = movie

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, mode='wb') as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, mode='wb') as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, mode='wb') as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, mode='wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, mode='rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, mode='rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, mode='rb') as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, mode='rb') as f:
            self.doc_lengths = pickle.load(f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    # docs = idx.get_document("merida")
    # print(f"firstdoc: {docs[0]}")


def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))


def bm25_command(query):
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query)



def bm25_tf_command(doc_id, term, b=BM25_B):
    idx = InvertedIndex()
    idx.load()
    print(idx.bm25_get_tf(doc_id, term, BM25_K1, b))


def tfidf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    # tf = idx.get_tf(doc_id, term)
    # idf = idx.get_idf(term)
    tfidf = idx.get_tfidf(term, doc_id)
    print(f"TF-IDF score of {term} in doc {doc_id} is {tfidf}.")


def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    print(f"{term=}, {idx.get_idf(term):.2f}")


def bm25_idf_command(term):
    idx = InvertedIndex()
    idx.load()
    print(f"{term=}, {idx.get_bm25_idf(term):.2f}")


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


def tokenize_text(text):
    text = clean_text(text)
    stop_words = load_stopwords()
    result = []

    def _filter(tokens):
        if token and token not in stop_words:
            return True
        return False

    for token in text.split():
        if _filter(token):
            token = stemmer.stem(token)
            result.append(token)

    return result


def is_matching(query_toks, movie_toks):
    for query_tok in query_toks:
        for movie_tok in movie_toks:
            if query_tok in movie_tok:
                return True
    return False


def search_movies(query, n_results=5):
    movies = load_movies()
    result = []
    idx = InvertedIndex()
    idx.load()
    seen, res = set(), []
    query_tokens = tokenize_text(query)
    for qt in query_tokens:
        matching_doc_ids = idx.get_document(qt)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            matching_doc = idx.docmap[matching_doc_id]
            result.append(matching_doc)
            if len(result) >= n_results:
                return result

    # query_tokens = tokenize_text(query)
    # for movie in movies:
    #     movie_tokens = tokenize_text(movie['title'])
    #     if is_matching(query_tokens, movie_tokens):
    #         result.append(movie)
    #     if len(result) == n_results:
    #         break
    return result
