#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest='command', help="available command")

    verify = subparsers.add_parser("verify", help="verify model")
    embed = subparsers.add_parser("embed_text", help="verify model")
    embed.add_argument("text", help="text to embed")
    verify_embedding = subparsers.add_parser("verify_embeddings", help="verify the embeddings")

    args = parser.parse_args()

    match args.command:
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
