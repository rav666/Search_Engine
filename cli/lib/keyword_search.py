from lib.search_utils import load_movies
def search_movies(query, n_results):
    movies = load_movies()
    result = []
    for movie in movies:
        if query.lower() in movie['title'].lower():
            result.append(movie)
        if len(result) == n_results:
            break
    return result
