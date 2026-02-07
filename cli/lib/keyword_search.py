import pickle

from lib.search_utils import load_movies,load_stopwords,CACHE_PATH
import string
from nltk.stem import PorterStemmer
from collections import defaultdict
import os
stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'
    def _add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
    def get_document(self, term):
        result = sorted(self.index[term])
        return list(result)

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

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()
    docs = idx.get_document("merida")
    print(f"firstdoc: {docs[0]}")



def clean_text(text):
    text= text.lower()
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

def search_movies(query, n_results):
    movies = load_movies()
    result = []
    query_tokens = tokenize_text(query)
    for movie in movies:
        movie_tokens = tokenize_text(movie['title'])
        if is_matching(query_tokens, movie_tokens):
            result.append(movie)
        if len(result) == n_results:
            break
    return result
