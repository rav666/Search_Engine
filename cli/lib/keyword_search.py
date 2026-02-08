import os
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from lib.search_utils import load_movies, load_stopwords, CACHE_PATH

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'
        self.term_frequencies = defaultdict(Counter)
        self.term_frequencies_path = CACHE_PATH / 'term_frequencies.pkl'

    def _add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_document(self, term):
        result = sorted(self.index[term])
        return list(result)

    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("can only have 1 tokens")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]

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

    def load(self):
        with open(self.index_path, mode='rb') as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, mode='rb') as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, mode='rb') as f:
            self.term_frequencies = pickle.load(f)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    # docs = idx.get_document("merida")
    # print(f"firstdoc: {docs[0]}")

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print (idx.get_tf(doc_id, term))



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
