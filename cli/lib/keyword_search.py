from lib.search_utils import load_movies, load_stopwords
import string
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
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
