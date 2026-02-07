from lib.search_utils import load_movies
import string

def clean_text(text):
    text= text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    return text


def tokenize_text(text):
    text = clean_text(text)
    tokens = [i for i in text.split() if i]
    return tokens

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
