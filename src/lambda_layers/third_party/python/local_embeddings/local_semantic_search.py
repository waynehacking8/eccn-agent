#!/usr/bin/env python3
"""
Local semantic search using Sharepoint Embedding Tool's vector search capabilities
"""

import pickle
import numpy as np
import sys
import os
from typing import List, Dict, Any, Optional

# Add the Sharepoint Embedding Tool path
sys.path.append('/home/wayneleo8/Sharepoint%20Embedding%20Tool/core')
sys.path.append('/home/wayneleo8/Sharepoint%20Embedding%20Tool/utils')

# Import the vector search functionality
from vector_search import VectorSearch, cosine_similarity
from openai_client import OpenAIClient

# Global variables
_embeddings_data = None
_entries = None
_openai_client = None

def load_eccn_embeddings():
    """Load ECCN embeddings from pickle file"""
    global _embeddings_data, _entries
    if _embeddings_data is None:
        pickle_path = '/home/wayneleo8/eccn-agent/src/sagemaker/eccn_embeddings.pkl'
        try:
            with open(pickle_path, 'rb') as f:
                _embeddings_data = pickle.load(f)
            _entries = _embeddings_data['entries']
            print(f" Loaded {len(_entries)} ECCN entries for local search")
        except Exception as e:
            print(f" Error loading embeddings: {e}")
            _embeddings_data = None
            _entries = []
    return _embeddings_data, _entries

def initialize_openai_client():
    """Initialize OpenAI client for query embedding"""
    global _openai_client
    if _openai_client is None:
        try:
            _openai_client = OpenAIClient()
            print(" OpenAI client initialized for query embedding")
        except Exception as e:
            print(f" Error initializing OpenAI client: {e}")
            _openai_client = None
    return _openai_client

def create_project_data_structure():
    """Create project data structure compatible with VectorSearch"""
    global _entries
    
    if not _entries:
        return None
    
    # Extract embeddings and create arrays
    embeddings = []
    texts = []
    metadata = []
    
    for entry in _entries:
        if 'embedding' in entry:
            embeddings.append(entry['embedding'])
            
            # Create text representation
            text_parts = []
            if entry.get('eccn_code'):
                text_parts.append(f"ECCN: {entry['eccn_code']}")
            if entry.get('description'):
                text_parts.append(entry['description'])
            if entry.get('extra_details'):
                text_parts.append(entry['extra_details'])
            
            text = ' | '.join(text_parts)
            texts.append(text)
            
            # Create metadata
            meta = {
                'eccn_code': entry.get('eccn_code', ''),
                'description': entry.get('description', ''),
                'extra_details': entry.get('extra_details', ''),
                'eccn_category': entry.get('eccn_category', ''),
                'product_group': entry.get('product_group', ''),
                'lastUpdated': entry.get('timestamp', '2024-01-01T00:00:00Z')
            }
            metadata.append(meta)
    
    if not embeddings:
        return None
    
    # Create project data structure
    project_data = {
        'embedArray': np.array(embeddings),
        'textArray': texts,
        'metaData': metadata,
        'embedMetaData': metadata  # Same as metadata for simplicity
    }
    
    return project_data

def local_semantic_search(query_text: str, size: int = 5, eccn_category: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform local semantic search using Sharepoint Embedding Tool's VectorSearch
    """
    try:
        # Load data
        load_eccn_embeddings()
        
        # Initialize OpenAI client for query embedding
        client = initialize_openai_client()
        if client is None:
            return {
                "total_hits": 0,
                "results": [],
                "error": "OpenAI client not available"
            }
        
        # Create project data structure
        project_data = create_project_data_structure()
        if project_data is None:
            return {
                "total_hits": 0,
                "results": [],
                "error": "No embedding data available"
            }
        
        # Generate query embedding
        try:
            query_embedding = np.array(client.generate_embeddings(query_text))
        except Exception as e:
            # If OpenAI fails, use cosine similarity with existing embeddings
            print(f"OpenAI query embedding failed: {e}")
            return fallback_search(query_text, size, eccn_category)
        
        # Set up filters
        filters = {}
        if eccn_category:
            filters['eccn_category'] = eccn_category
        
        # Create VectorSearch instance
        vector_search = VectorSearch(
            query_embedding=query_embedding,
            project_data=project_data,
            top_k=size,
            similarity_threshold=0.3,  # Lower threshold for more results
            filters=filters
        )
        
        # Perform search
        search_results = vector_search.search()
        
        # Format results
        results = []
        for result in search_results:
            metadata = result['metadata']
            results.append({
                'eccn_code': metadata.get('eccn_code', ''),
                'description': metadata.get('description', ''),
                'extra_details': metadata.get('extra_details', ''),
                'score': float(result['similarity_score'])
            })
        
        return {
            "total_hits": len(search_results),
            "results": results
        }
        
    except Exception as e:
        print(f"Error in local semantic search: {e}")
        return {
            "total_hits": 0,
            "results": [],
            "error": str(e)
        }

def fallback_search(query_text: str, size: int = 5, eccn_category: Optional[str] = None) -> Dict[str, Any]:
    """
    Fallback search using simple text matching when OpenAI is not available
    """
    global _entries
    
    if not _entries:
        return {"total_hits": 0, "results": []}
    
    # Simple text matching
    query_lower = query_text.lower()
    matches = []
    
    for entry in _entries:
        score = 0
        
        # Check ECCN category filter
        if eccn_category and not entry.get('eccn_code', '').startswith(eccn_category):
            continue
        
        # Simple text matching
        if query_lower in entry.get('description', '').lower():
            score += 0.8
        if query_lower in entry.get('extra_details', '').lower():
            score += 0.6
        if query_lower in entry.get('eccn_code', '').lower():
            score += 0.9
        
        # Check for individual words
        words = query_lower.split()
        for word in words:
            if word in entry.get('description', '').lower():
                score += 0.1
            if word in entry.get('extra_details', '').lower():
                score += 0.1
        
        if score > 0:
            matches.append({
                'entry': entry,
                'score': score
            })
    
    # Sort by score
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    # Format results
    results = []
    for match in matches[:size]:
        entry = match['entry']
        results.append({
            'eccn_code': entry.get('eccn_code', ''),
            'description': entry.get('description', ''),
            'extra_details': entry.get('extra_details', ''),
            'score': match['score']
        })
    
    return {
        "total_hits": len(matches),
        "results": results
    }

def test_local_search():
    """Test the local semantic search"""
    print(" Testing Local Semantic Search with Sharepoint Tool")
    print("=" * 60)
    
    # Test 1: Basic search
    print("\n Test 1: Basic Search")
    results = local_semantic_search("encryption software", size=3)
    
    if results['results']:
        print(f" Found {len(results['results'])} results:")
        for i, result in enumerate(results['results'], 1):
            print(f"  {i}. {result['eccn_code']}: {result['description'][:80]}...")
            print(f"     Score: {result['score']:.4f}")
    else:
        print(f" No results found. Error: {results.get('error', 'Unknown')}")
    
    # Test 2: Category filter
    print("\n Test 2: Category Filter")
    results = local_semantic_search("telecommunications", size=3, eccn_category="5")
    
    if results['results']:
        print(f" Found {len(results['results'])} results in category 5:")
        for i, result in enumerate(results['results'], 1):
            print(f"  {i}. {result['eccn_code']}: {result['description'][:80]}...")
            print(f"     Score: {result['score']:.4f}")
    else:
        print(f" No results found. Error: {results.get('error', 'Unknown')}")

if __name__ == "__main__":
    test_local_search()