# Sample Datasets

This directory contains small, reproducible datasets for Lumina Memory System demos, testing, and validation.

## Available Datasets

### AI Research Papers (i_papers_sample.json)

A curated collection of 20 influential AI research papers with:
- **Document count**: 20 papers
- **Query count**: 10 test queries  
- **Categories**: NLP, Computer Vision, Reinforcement Learning, Generative Models, etc.
- **Purpose**: End-to-end testing, semantic search validation, performance benchmarking

**Structure**:
`json
{
  "metadata": {...},
  "documents": [
    {
      "id": "doc_001",
      "title": "Attention Is All You Need",
      "content": "The Transformer architecture revolutionized...",
      "metadata": {
        "category": "nlp",
        "year": 2017,
        "authors": ["Vaswani", "Shazeer", "Parmar"],
        "venue": "NeurIPS"
      }
    }
  ],
  "test_queries": [
    {
      "id": "query_001", 
      "query": "What are transformer neural networks?",
      "expected_categories": ["nlp"],
      "expected_docs": ["doc_001", "doc_002", "doc_003"]
    }
  ]
}
`

## Usage

### Loading in Python
`python
from data.sample.datasets import load_ai_papers_sample

# Load documents and queries
docs, queries = load_ai_papers_sample()

# Create text corpus for ingestion
from data.sample.datasets import create_test_corpus
corpus = create_test_corpus(docs, include_metadata=True)

# Ingest into memory system
memory_system.ingest(corpus)
`

### Validation
`python
from data.sample.datasets import validate_dataset

# Validate dataset consistency
report = validate_dataset("ai_papers")
if report['valid']:
    print(f" Dataset valid: {report['statistics']}")
else:
    print(f" Issues found: {report['issues']}")
`

### End-to-End Testing
`ash
# Run E2E tests with sample data
pytest tests/test_e2e_sample_dataset.py -v

# Quick validation
python data/sample/datasets.py
`

## Dataset Guidelines

When adding new sample datasets:

1. **Keep them small**: 10-50 documents maximum
2. **Include metadata**: Category, source, relevant tags
3. **Provide test queries**: With expected document matches
4. **Add validation**: Check consistency and completeness
5. **Document purpose**: What the dataset is designed to test

## Performance Characteristics

The AI papers dataset is designed for:
- **Ingestion**: ~1-2 seconds for 20 documents
- **Query latency**: <100ms per query on average
- **Memory usage**: <50MB total
- **Semantic coverage**: Multiple domains and query types

Perfect for CI/CD pipelines and development validation!
