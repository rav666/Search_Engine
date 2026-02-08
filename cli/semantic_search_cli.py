#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest='command', help="available command")

    verifying = subparsers.add_parser("verify", help="verify the model with its features")
    embed = subparsers.add_parser("embed_text", help="prototype of embedding query")
    embed.add_argument("text", help="text to embed (PROTOTYPE)")
    verify_embedding = subparsers.add_parser("verify_embeddings", help="LOADING OR CREATING EMBEDDINGS")
    embed_query = subparsers.add_parser("embedquery", help="MAKING EMBEDDINGS FOR THE SEARCH TEXT")
    embed_query.add_argument("query", type=str, help="TEXT U WANT TO SEARCH IN MOVIES")
    args = parser.parse_args()

    match args.command:
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case 'verify':
            verify_model()
        case 'embed_text':

            embed_text(args.text)

        case _:
            parser.print_help()


if __name__ == '__main__':
    main()
