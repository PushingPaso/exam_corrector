"""
Test script to verify the setup is working correctly.
Run this after installing dependencies to check everything is configured.
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        import langchain
        print("✓ langchain")
    except ImportError as e:
        print(f"✗ langchain: {e}")
        return False
    
    try:
        import langchain_openai
        print("✓ langchain-openai")
    except ImportError as e:
        print(f"✗ langchain-openai: {e}")
        return False
    
    try:
        import langchain_huggingface
        print("✓ langchain-huggingface")
    except ImportError as e:
        print(f"✗ langchain-huggingface: {e}")
        return False
    
    try:
        import sentence_transformers
        print("✓ sentence-transformers")
    except ImportError as e:
        print(f"✗ sentence-transformers: {e}")
        return False
    
    try:
        import torch
        print(f"✓ torch (version: {torch.__version__})")
    except ImportError as e:
        print(f"✗ torch: {e}")
        return False
    
    try:
        from exam.rag import huggingface_embeddings
        print("✓ exam.rag module")
    except ImportError as e:
        print(f"✗ exam.rag module: {e}")
        return False
    
    try:
        from exam.openai import AIOracle
        print("✓ exam.openai module")
    except ImportError as e:
        print(f"✗ exam.openai module: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_embeddings():
    """Test that embeddings can be created and used."""
    print("\n" + "=" * 60)
    print("Testing embeddings...")
    print("=" * 60)
    
    try:
        from exam.rag import huggingface_embeddings
        
        print("\nCreating embeddings model (small)...")
        embeddings = huggingface_embeddings("small")
        
        print("\nTesting embedding generation...")
        test_text = "This is a test sentence for embedding."
        embedding = embeddings.embed_query(test_text)
        
        print(f"✓ Generated embedding with {len(embedding)} dimensions")
        print(f"  First 5 values: {embedding[:5]}")
        
        return True
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_openrouter_config():
    """Test OpenRouter API key configuration."""
    print("\n" + "=" * 60)
    print("Testing OpenRouter configuration...")
    print("=" * 60)
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not api_key:
        print("⚠ OPENROUTER_API_KEY not set in environment")
        print("  This is OK - the system will ask for it when needed")
        print("  To set it permanently:")
        print("    export OPENROUTER_API_KEY='your-key-here'")
        return True
    
    if api_key.startswith("sk-or-v1-"):
        print(f"✓ OPENROUTER_API_KEY is set")
        print(f"  Key starts with: {api_key[:20]}...")
        return True
    else:
        print(f"⚠ OPENROUTER_API_KEY is set but doesn't look like a valid key")
        print(f"  Expected format: sk-or-v1-...")
        print(f"  Found: {api_key[:20]}...")
        return True


def test_vector_store():
    """Test vector store creation (without filling it)."""
    print("\n" + "=" * 60)
    print("Testing vector store...")
    print("=" * 60)
    
    try:
        from exam.rag import sqlite_vector_store
        from exam import DIR_ROOT
        import tempfile
        import os
        
        # Use a temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            tmp_db = tmp.name
        
        try:
            print(f"\nCreating test vector store at: {tmp_db}")
            store = sqlite_vector_store(db_file=tmp_db, model="small")
            print("✓ Vector store created successfully")
            
            # Test adding and searching
            print("\nTesting add and search...")
            test_texts = [
                "Software engineering is about building reliable systems.",
                "Testing is an important part of quality assurance.",
            ]
            store.add_texts(test_texts)
            print(f"✓ Added {len(test_texts)} test documents")
            
            results = store.similarity_search("software quality", k=1)
            print(f"✓ Search returned {len(results)} results")
            if results:
                print(f"  Top result: {results[0].page_content[:50]}...")
            
            return True
        finally:
            # Clean up temp database
            if os.path.exists(tmp_db):
                os.unlink(tmp_db)
                print(f"\n✓ Cleaned up test database")
    
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("EXAM CORRECTOR - SETUP VERIFICATION")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    if results[-1][1]:  # Only continue if imports work
        results.append(("Embeddings", test_embeddings()))
        results.append(("OpenRouter Config", test_openrouter_config()))
        results.append(("Vector Store", test_vector_store()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✓ All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("  1. Set OPENROUTER_API_KEY environment variable")
        print("  2. Run: python -m exam.rag --fill")
        print("  3. Run: python -m exam.solution")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        print("\nCommon fixes:")
        print("  - Run: pip install -r requirements.txt")
        print("  - Check Python version (needs 3.8+)")
        print("  - Ensure you have ~2GB free disk space for models")
        return 1


if __name__ == "__main__":
    sys.exit(main())