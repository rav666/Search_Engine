#!/usr/bin/env python3

import argparse

from lib.keyword_search import search_movies, build_command, tf_command, idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description='Keyword Search CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    search_parser = subparsers.add_parser('search', help='Search movies')
    search_parser.add_argument('query', type=str, help='Search Query')
    build_parser = subparsers.add_parser('build', help='Just build it')

    tf_parser = subparsers.add_parser('tf', help='Calculate term frequency')
    tf_parser.add_argument('id', type=int, help='DOC ID')
    tf_parser.add_argument('term', type=str, help='WORD WHOSE FREQUENCY U WANT')
    idf_parser = subparsers.add_parser('idf', help='IDF')
    idf_parser.add_argument('term', type=str, help='Inverse Document Frequency')

    args = parser.parse_args()
    match args.command:
        case 'search':

            print(f"Searching for: {args.query}")
            results = search_movies(args.query, 5)
            for i, result in enumerate(results):
                print(f"{i + 1}, {result['title']}")
        case 'build':
            build_command()
        case 'tf':
            tf_command(args.id, args.term)
        case 'idf':
            idf_command(args.term)
        case _:
            print('Searching movies using Google Search')
            # parser.print_help()


if __name__ == '__main__':
    main()
