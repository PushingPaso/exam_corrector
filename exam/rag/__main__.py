from exam.rag import *
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='RAG Vector Store Manager')
    parser.add_argument('--fill', action='store_true',
                        help='Fill vector store with course slides')
    parser.add_argument('--model', type=str, default='bge-base',
                        help='Embedding model to use (default: bge-base)')
    parser.add_argument('--force', action='store_true',
                        help='Force recreate database (deletes existing)')

    args = parser.parse_args()

    if args.fill:
        print(f"# Creating vector store with model: {args.model}")

        # Force recreate if requested
        if args.force:
            import os
            if os.path.exists(FILE_DB):
                print(f"# Deleting existing database: {FILE_DB}")
                os.remove(FILE_DB)

        # Create vector store with specified model
        vector_store = sqlite_vector_store(model=args.model)

        print(f"# Vector store created at {FILE_DB}")
        print(f"# Filling with course slides...")

        slide_count = 0
        for slide in all_slides():
            vector_store.add_texts(
                texts=[slide.content],
                metadatas=[{
                    "source": slide.source,
                    "lines": slide.lines,
                    "index": slide.index
                }],
            )
            slide_count += 1
            if slide_count % 10 == 0:
                print(f"# Added {slide_count} slides...")

        print(f"# Vector store filled successfully with {slide_count} slides")

    else:
        # Query mode
        vector_store = sqlite_vector_store(model=args.model)

        print(f"# Vector store loaded successfully: it contains {vector_store.get_dimensionality()} dimensions.")

        while True:
            try:
                query = input("Enter your query (or 'exit' to quit):\n\t")
                if query.strip().lower() == 'exit':
                    break

                results = vector_store.similarity_search(query, k=3)

                for i, doc in enumerate(results, 1):
                    print(f"\n\t# Result {i} from: {doc.metadata['source']}")
                    print("\t\t" + doc.page_content[:200].replace("\n", "\n\t\t"))
                    if len(doc.page_content) > 200:
                        print("\t\t...")
                    print("\t\t---")

            except (EOFError, KeyboardInterrupt):
                break

    print("# Goodbye!")


if __name__ == "__main__":
    main()