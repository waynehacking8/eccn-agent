
"""
ECCN Search Functions for Optimization Notebook

These functions provide OpenSearch replacement functionality using local pickle embeddings.
"""

import pickle
import numpy as np
from typing import Dict, List, Any, Optional
import os
import sys

# Add the Sharepoint Embedding Tool path for query embeddings
from pathlib import Path
current_dir = Path(__file__).parent
sharepoint_tool_path = current_dir.parent.parent.parent / "Sharepoint%20Embedding%20Tool" / "utils"
if sharepoint_tool_path.exists():
    sys.path.append(str(sharepoint_tool_path))

try:
    from openai_client import OpenAIClient
    _openai_client = None  # Will be initialized when needed
except ImportError:
    print("Warning: Could not import OpenAI client")
    _openai_client = None

# Global variables for loaded data
_embeddings_data = None
_entries = None
_embeddings = None

def load_eccn_embeddings(pickle_path: str = None):
    """Load ECCN embeddings from pickle file"""
    global _embeddings_data, _entries, _embeddings
    
    if _embeddings_data is None:
        if pickle_path is None:
            # Try multiple possible paths for the pickle file
            current_dir = Path(__file__).parent
            possible_paths = [
                current_dir / "eccn_embeddings.pkl",  # Same directory as this script
                Path("/opt/python/local_embeddings/eccn_embeddings.pkl"),  # Lambda layer path
                Path("/home/wayneleo8/eccn-agent/src/sagemaker/eccn_embeddings.pkl"),  # Local development path
            ]
            
            pickle_path = None
            for path in possible_paths:
                if path.exists():
                    pickle_path = path
                    break
            
            if pickle_path is None:
                raise FileNotFoundError(f"ECCN embeddings file not found in any of: {[str(p) for p in possible_paths]}")
            
        print(f"Loading ECCN embeddings from: {pickle_path}")
        
        try:
            with open(pickle_path, 'rb') as f:
                _embeddings_data = pickle.load(f)
            
            _entries = _embeddings_data["entries"]
            _embeddings = [entry["embedding"] for entry in _entries]
            
            print(f"Loaded {len(_entries)} ECCN entries with embeddings")
            print(f" Embedding dimension: {_embeddings_data['metadata']['embedding_dimension']}")
        except Exception as e:
            print(f"Error loading ECCN embeddings: {e}")
            raise
    
    return _embeddings_data

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    return dot_product / norms if norms != 0 else 0.0

def semantic_search(query_text: str, size: int = 5, eccn_category: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform semantic search using embeddings
    
    Args:
        query_text: Text to search for
        size: Number of results to return
        eccn_category: Optional ECCN category filter (e.g., '3', '4', '5')
        
    Returns:
        Dictionary with search results in OpenSearch format
    """
    # Load data if not already loaded
    load_eccn_embeddings()
    
    # Initialize OpenAI client if not already done
    global _openai_client
    if _openai_client is None:
        try:
            _openai_client = OpenAIClient()
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            return {"total_hits": 0, "results": []}
    
    try:
        # Generate embedding for query
        query_embedding = _openai_client.generate_embeddings(query_text)
        
        # Calculate similarities
        similarities = []
        for i, entry in enumerate(_entries):
            # Apply category filter if specified
            if eccn_category and not entry.get("eccn_category", "").startswith(eccn_category):
                continue
            
            similarity = cosine_similarity(query_embedding, entry["embedding"])
            similarities.append({
                "index": i,
                "similarity": similarity,
                "entry": entry
            })
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = similarities[:size]
        
        # Format results to match OpenSearch format
        results = []
        for result in top_results:
            entry = result["entry"]
            formatted_result = {
                "eccn_code": entry.get("eccn_code", ""),
                "description": entry.get("description", ""),
                "extra_details": entry.get("details", ""),
                "score": result["similarity"]
            }
            results.append(formatted_result)
        
        return {
            "total_hits": len(similarities),
            "results": results
        }
        
    except Exception as e:
        print(f"Error in semantic search: {e}")
        return {"total_hits": 0, "results": []}

def find_eccn_info_by_eccn_code(eccn_code: str, size: int = 5) -> Dict[str, Any]:
    """
    Find ECCN information by exact code match
    
    Args:
        eccn_code: ECCN code to search for
        size: Maximum number of results to return
        
    Returns:
        Dictionary with search results in OpenSearch format
    """
    # Handle EAR99 special case
    if eccn_code == "EAR99":
        return {
            "total_hits": 1,
            "results": [
                {
                    "eccn_code": "EAR99",
                    "description": "Items subject to the EAR but not listed on the Commerce Control List. Generally consist of low-technology consumer goods and do not require a license in most situations.",
                    "extra_details": "",
                    "score": 1.0
                }
            ]
        }
    
    # Load data if not already loaded
    load_eccn_embeddings()
    
    # Search for exact matches
    results = []
    for entry in _entries:
        if entry.get("eccn_code", "").upper() == eccn_code.upper():
            formatted_result = {
                "eccn_code": entry.get("eccn_code", ""),
                "description": entry.get("description", ""),
                "extra_details": entry.get("details", ""),
                "score": 1.0
            }
            results.append(formatted_result)
            
            if len(results) >= size:
                break
    
    return {
        "total_hits": len(results),
        "results": results
    }

def get_unique_categories() -> Dict[str, Any]:
    """Get unique ECCN categories"""
    load_eccn_embeddings()
    
    categories = {}
    for entry in _entries:
        category = entry.get("eccn_category", "")
        if category and category not in categories:
            categories[category] = {
                "eccn_category": category,
                "eccn_category_detail": entry.get("category_description", "")
            }
    
    categories_list = list(categories.values())
    
    return {
        "total_categories": len(categories_list),
        "category_list": categories_list
    }

def get_product_groups_by_category(eccn_category: str) -> Dict[str, Any]:
    """Get product groups for a specific category"""
    load_eccn_embeddings()
    
    product_groups = {}
    for entry in _entries:
        if entry.get("eccn_category", "").startswith(eccn_category):
            group_name = entry.get("product_group_name", "")
            if group_name and group_name not in product_groups:
                product_groups[group_name] = {
                    "group_product": group_name,
                    "group_product_detail": entry.get("product_group_description", "")
                }
    
    groups_list = list(product_groups.values())
    
    return {
        "total_group": len(groups_list),
        "group_name_list": groups_list
    }

def find_eccn_details_with_category_production_group(eccn_category: str, product_group: str, size: int = 10) -> Dict[str, Any]:
    """
    Search with filtering options for ECCN category and product group
    
    Args:
        eccn_category: ECCN category to filter results
        product_group: Product group to filter results
        size: Maximum number of results to return
        
    Returns:
        Dictionary with search results
    """
    load_eccn_embeddings()
    
    results = []
    for entry in _entries:
        # Check category filter
        if eccn_category and not entry.get("eccn_category", "").startswith(eccn_category):
            continue
        
        # Check product group filter
        if product_group and product_group.lower() not in entry.get("product_group_name", "").lower():
            continue
        
        formatted_result = {
            "eccn_code": entry.get("eccn_code", ""),
            "description": entry.get("description", ""),
            "extra_details": entry.get("details", ""),
            "score": 1.0
        }
        results.append(formatted_result)
        
        if len(results) >= size:
            break
    
    return {
        "total_hits": len(results),
        "results": results
    }

def get_tools_definition():
    """Get tool definitions for compatibility"""
    return [
        {
            "toolSpec": {
                "name": "semantic_search",
                "description": "Search for ECCN records using semantic similarity matching",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": "The technical description to search for"
                            },
                            "size": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)"
                            },
                            "eccn_category": {
                                "type": "string",
                                "description": "Optional ECCN category filter (e.g., '5', '3')"
                            }
                        },
                        "required": ["query_text"]
                    }
                }
            }
        },
        {
            "toolSpec": {
                "name": "find_eccn_info_by_eccn_code",
                "description": "Look up an ECCN record by code",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "eccn_code": {
                                "type": "string",
                                "description": "The ECCN code to look up"
                            },
                            "size": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)"
                            }
                        },
                        "required": ["eccn_code"]
                    }
                }
            }
        }
    ]

# Test function
def test_functions():
    """Test the search functions"""
    print("Testing ECCN search functions...")
    
    # Test semantic search
    results = semantic_search("encryption software", size=3)
    print(f"Semantic search: {len(results['results'])} results found")
    
    # Test ECCN lookup
    results = find_eccn_info_by_eccn_code("5A002")
    print(f"ECCN lookup: {len(results['results'])} results found")
    
    # Test categories
    categories = get_unique_categories()
    print(f"Categories: {categories['total_categories']} found")
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    test_functions()
