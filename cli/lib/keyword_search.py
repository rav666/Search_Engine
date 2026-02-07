from lib.search_utils import load_movies
def search_movies(query, n_results):
    movies = load_movies()
    result = []
    for movie in movies:
        if query in movie['title']:
            result.append(movie)
        if len(result) == n_results:
            break
    return result
