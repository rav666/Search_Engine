from lib.search_utils import load_movies
import string

def clean_text(text):
    text= text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    return text
def search_movies(query, n_results):
    movies = load_movies()
    result = []
    query = clean_text(query)
    for movie in movies:
        if query.lower() in clean_text(movie['title']):
            result.append(movie)
        if len(result) == n_results:
            break
    return result
