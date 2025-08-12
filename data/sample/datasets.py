"""
Sample dataset utilities for Lumina Memory System.

This module provides utilities for loading and working with the sample datasets
included in data/sample/. These datasets are designed for:

- Demos and tutorials
- End-to-end testing 
- Performance benchmarking
- Integration validation

Usage:
    from lumina_memory.datasets import load_ai_papers_sample
    
    docs, queries = load_ai_papers_sample()
    memory_system.ingest([doc['content'] for doc in docs])
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib.util

def get_sample_data_path() -> Path:
    """Get the path to the sample data directory."""
    # Try to find the data directory relative to this module
    current_dir = Path(__file__).parent
    
    # Look for data/sample from the package root
    for parent in [current_dir, current_dir.parent, current_dir.parent.parent]:
        data_path = parent / "data" / "sample"
        if data_path.exists():
            return data_path
    
    # Fallback: assume we're in a standard package structure
    return Path(__file__).parent.parent.parent / "data" / "sample"

def load_ai_papers_sample() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load the AI research papers sample dataset.
    
    Returns:
        Tuple of (documents, test_queries)
        
    Documents format:
        {
            'id': str,
            'title': str, 
            'content': str,
            'metadata': dict
        }
        
    Queries format:
        {
            'id': str,
            'query': str,
            'expected_categories': List[str],
            'expected_docs': List[str]
        }
    """
    data_path = get_sample_data_path() / "ai_papers_sample.json"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Sample dataset not found at {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['documents'], data['test_queries']

def load_sample_for_testing(dataset_name: str = "ai_papers") -> Dict[str, Any]:
    """
    Load a sample dataset formatted for testing.
    
    Args:
        dataset_name: Name of the dataset to load
        
    Returns:
        Dictionary with keys: documents, queries, metadata
    """
    if dataset_name == "ai_papers":
        docs, queries = load_ai_papers_sample()
        
        # Load metadata
        data_path = get_sample_data_path() / "ai_papers_sample.json"
        with open(data_path, 'r') as f:
            full_data = json.load(f)
        
        return {
            'documents': docs,
            'queries': queries, 
            'metadata': full_data['metadata']
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def create_test_corpus(docs: List[Dict[str, Any]], 
                      include_metadata: bool = True) -> List[str]:
    """
    Create a text corpus from document list for ingestion.
    
    Args:
        docs: List of document dictionaries
        include_metadata: Whether to include metadata in the text
        
    Returns:
        List of text strings ready for ingestion
    """
    corpus = []
    for doc in docs:
        text = doc['content']
        
        if include_metadata and 'title' in doc:
            text = f"Title: {doc['title']}\n\n{text}"
            
        if include_metadata and 'metadata' in doc:
            meta = doc['metadata']
            if 'category' in meta:
                text = f"Category: {meta['category']}\n{text}"
            if 'authors' in meta:
                authors = ', '.join(meta['authors']) if isinstance(meta['authors'], list) else meta['authors']
                text = f"Authors: {authors}\n{text}"
        
        corpus.append(text)
    
    return corpus

def validate_dataset(dataset_name: str = "ai_papers") -> Dict[str, Any]:
    """
    Validate a sample dataset for completeness and consistency.
    
    Returns:
        Validation report with statistics and any issues found
    """
    data = load_sample_for_testing(dataset_name)
    docs = data['documents']
    queries = data['queries']
    metadata = data['metadata']
    
    report = {
        'dataset_name': dataset_name,
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    # Check document consistency
    doc_ids = set(doc['id'] for doc in docs)
    if len(doc_ids) != len(docs):
        report['issues'].append("Duplicate document IDs found")
        report['valid'] = False
    
    # Check required fields
    for i, doc in enumerate(docs):
        required_fields = ['id', 'content']
        missing = [field for field in required_fields if field not in doc]
        if missing:
            report['issues'].append(f"Document {i} missing fields: {missing}")
            report['valid'] = False
    
    # Check query consistency
    for query in queries:
        if 'expected_docs' in query:
            expected_ids = set(query['expected_docs'])
            missing_ids = expected_ids - doc_ids
            if missing_ids:
                report['issues'].append(
                    f"Query {query['id']} references missing docs: {missing_ids}"
                )
                report['valid'] = False
    
    # Compute statistics
    report['statistics'] = {
        'document_count': len(docs),
        'query_count': len(queries),
        'avg_content_length': sum(len(doc['content']) for doc in docs) / len(docs),
        'categories': list(set(
            doc.get('metadata', {}).get('category') 
            for doc in docs 
            if doc.get('metadata', {}).get('category')
        )),
        'total_expected_matches': sum(
            len(query.get('expected_docs', [])) 
            for query in queries
        )
    }
    
    return report

# Example usage and testing functions
def run_basic_demo():
    """Run a basic demonstration using the sample dataset."""
    try:
        docs, queries = load_ai_papers_sample()
        print(f" Loaded {len(docs)} documents and {len(queries)} queries")
        
        # Show sample document
        sample_doc = docs[0]
        print(f"\nSample document:")
        print(f"Title: {sample_doc['title']}")
        print(f"Content: {sample_doc['content'][:100]}...")
        
        # Show sample query
        sample_query = queries[0]
        print(f"\nSample query:")
        print(f"Query: {sample_query['query']}")
        print(f"Expected docs: {sample_query['expected_docs']}")
        
        return True
    except Exception as e:
        print(f" Demo failed: {e}")
        return False

if __name__ == "__main__":
    # Run validation and demo
    print(" Validating sample dataset...")
    report = validate_dataset()
    
    if report['valid']:
        print(" Dataset validation passed")
        print(f" Statistics: {report['statistics']}")
    else:
        print(" Dataset validation failed")
        for issue in report['issues']:
            print(f"  - {issue}")
    
    print("\n Running basic demo...")
    run_basic_demo()
