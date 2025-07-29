import json
import os
import sys
from typing import Dict, List, Any, Optional

# Add the path to the search functions
from pathlib import Path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import the local search functions
from eccn_search_functions import (
    semantic_search as local_semantic_search,
    find_eccn_info_by_eccn_code as local_find_eccn_info_by_eccn_code,
    get_unique_categories as local_get_unique_categories,
    get_product_groups_by_category as local_get_product_groups_by_category,
    find_eccn_details_with_category_production_group as local_find_eccn_details_with_category_production_group,
)

# Load the local embeddings once at module level
print("Loading local ECCN embeddings...")
try:
    # Initialize the local search functions by calling them once
    local_get_unique_categories()
    print(" Local ECCN embeddings loaded successfully")
except Exception as e:
    print(f" Error loading local ECCN embeddings: {e}")
    print("Please ensure the embeddings pickle file exists at /home/wayneleo8/eccn-agent/src/sagemaker/eccn_embeddings.pkl")

def semantic_search(query_text: str, size: int = 5, eccn_category: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform semantic search using local embeddings with enhanced vector search
    
    Args:
        query_text: Text to search for
        size: Maximum number of results to return
        eccn_category: Optional ECCN category to filter results
        
    Returns:
        dict: Search results in OpenSearch-compatible format
    """
    try:
        # Try the enhanced local semantic search first
        from local_semantic_search import local_semantic_search as enhanced_search
        results = enhanced_search(query_text, size, eccn_category)
        
        # If that fails, fall back to the original function
        if not results.get('results') and 'error' in results:
            print(f"Enhanced search failed: {results['error']}")
            print("Falling back to original semantic search...")
            results = local_semantic_search(query_text, size, eccn_category)
        
        return results
        
    except Exception as e:
        print(f"Error in enhanced semantic search: {e}")
        # Final fallback to original function
        try:
            results = local_semantic_search(query_text, size, eccn_category)
            return results
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return {
                "total_hits": 0,
                "results": [],
                "error": str(e)
            }


def find_eccn_info_by_eccn_code(eccn_code: str, size: int = 5) -> Dict[str, Any]:
    """
    Find ECCN information by ECCN code using local embeddings
    
    Args:
        eccn_code: ECCN code to search for
        size: Maximum number of results to return
        
    Returns:
        dict: Search results in OpenSearch-compatible format
    """
    try:
        # Call the local ECCN lookup function
        results = local_find_eccn_info_by_eccn_code(eccn_code, size)
        
        # Results are already in the correct format
        return results
        
    except Exception as e:
        print(f"Error in local ECCN lookup: {e}")
        return {
            "total_hits": 0,
            "results": [],
            "error": str(e)
        }


def get_unique_categories() -> Dict[str, Any]:
    """
    Get all unique ECCN categories from local embeddings
    
    Returns:
        dict: List of unique ECCN categories
    """
    try:
        # Call the local categories function
        results = local_get_unique_categories()
        
        # Results are already in the correct format
        return results
        
    except Exception as e:
        print(f"Error in local categories retrieval: {e}")
        return {
            "total_categories": 0,
            "category_list": [],
            "error": str(e)
        }


def get_product_groups_by_category(eccn_category: str) -> Dict[str, Any]:
    """
    Get product groups for a specific ECCN category using local embeddings
    
    Args:
        eccn_category: ECCN category to filter results
        
    Returns:
        dict: Product groups for the category
    """
    try:
        # Call the local product groups function
        results = local_get_product_groups_by_category(eccn_category)
        
        # Results are already in the correct format
        return results
        
    except Exception as e:
        print(f"Error in local product groups retrieval: {e}")
        return {
            "total_group": 0,
            "group_name_list": [],
            "error": str(e)
        }


def find_eccn_details_with_category_production_group(
    eccn_category: str,
    product_group: str,
    size: int = 10,
) -> Dict[str, Any]:
    """
    Search with filtering options for ECCN category and product group using local embeddings
    
    Args:
        eccn_category: ECCN category to filter results
        product_group: product group to filter results
        size: Maximum number of results to return
        
    Returns:
        dict: Search results
    """
    try:
        # Call the local filtered search function
        results = local_find_eccn_details_with_category_production_group(
            eccn_category, product_group, size
        )
        
        # Results are already in the correct format
        return results
        
    except Exception as e:
        print(f"Error in local filtered search: {e}")
        return {
            "total_hits": 0,
            "results": [],
            "error": str(e)
        }


def get_tools_definition():
    """
    Get the tools definition for Claude to use with local embeddings
    
    Returns:
        list: Tool definitions compatible with the original OpenSearch tools
    """
    return [
        {
            "toolSpec": {
                "name": "semantic_search",
                "description": """Search for ECCN records using semantic similarity matching against official ECCN descriptions.
    
    This function uses local embeddings to find the most relevant ECCN classification based on
    technical descriptions, capabilities, or characteristics of a product. The search uses 
    semantic similarity to match your query against official ECCN descriptions as defined 
    by US export control regulations.
                """,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": """The technical description or characteristics to match against ECCN definitions.
        Do NOT include product names, marketing terms, or serial number - only include technical capabilities
        and specifications that would be relevant for export control classification.""",
                            },
                            "size": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                            },
                            "eccn_category": {
                                "type": "string",
                                "description": "Optional ECCN category to filter results (e.g., '5', '3')",
                            },
                        },
                        "required": ["query_text"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "find_eccn_info_by_eccn_code",
                "description": "Look up an ECCN record by the five digit eccn code using local embeddings. Use this when the query contains a specific ECCN code like '5A002' or '3A001'.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "eccn_code": {
                                "type": "string",
                                "description": "The ECCN code to look up (e.g., '5A002', '3A001')",
                            },
                            "size": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                            },
                        },
                        "required": ["eccn_code"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_unique_categories",
                "description": "Get all unique ECCN categories from the local embeddings database.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_product_groups_by_category",
                "description": "Get all unique product groups for a given ECCN category from local embeddings.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "eccn_category": {
                                "type": "string",
                                "description": "ECCN category to filter results (e.g., '3', '4', '5')",
                            }
                        },
                        "required": ["eccn_category"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "find_eccn_details_with_category_production_group",
                "description": "Search with filtering options for ECCN category and product group using local embeddings.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "eccn_category": {
                                "type": "string",
                                "description": "ECCN category to filter results (e.g., '5', '3')",
                            },
                            "product_group": {
                                "type": "string",
                                "description": "product group to filter results",
                            },
                            "size": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)",
                            },
                        },
                        "required": ["eccn_category", "product_group"],
                    }
                },
            }
        },
    ]


# Test the local functions on import
if __name__ == "__main__":
    print("Testing local ECCN search functions...")
    
    # Test semantic search
    try:
        results = semantic_search("encryption software", size=3)
        print(f" Semantic search: {len(results['results'])} results found")
    except Exception as e:
        print(f" Semantic search error: {e}")
    
    # Test ECCN lookup
    try:
        results = find_eccn_info_by_eccn_code("5A002")
        print(f" ECCN lookup: {len(results['results'])} results found")
    except Exception as e:
        print(f" ECCN lookup error: {e}")
    
    # Test categories
    try:
        results = get_unique_categories()
        print(f" Categories: {results['total_categories']} categories found")
    except Exception as e:
        print(f" Categories error: {e}")
    
    print("Local ECCN search functions test completed!")