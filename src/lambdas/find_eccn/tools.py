import json
import os

import boto3
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth, OpenSearch

ECCN_SOURCES = ["eccn_code", "description", "extra_details", "timestamp"]

# Singleton instance for OpenSearch client
_opensearch_client = None


def format_search_result(source, hit):
    """
    Format a single search result entry

    Args:
        source (dict): The _source data from OpenSearch
        hit (dict): The hit data containing the score

    Returns:
        dict: Formatted search result
    """
    return {
        "eccn_code": source.get("eccn_code", ""),
        "description": source.get("description", ""),
        "extra_details": source.get("extra_details", ""),
        "score": hit.get("_score", 0),
    }


def get_opensearch_client():
    """
    Create and return an OpenSearch client using environment variables
    This is implemented as a singleton pattern to avoid creating multiple clients.

    This function creates an authenticated connection to the OpenSearch
    serverless collection using AWS SigV4 authentication. It reads the
    endpoint from environment variables.

    Returns:
        OpenSearch: Configured OpenSearch client ready to use

    Environment Variables:
        OPENSEARCH_ENDPOINT: The host endpoint for OpenSearch
        AWS_REGION: The AWS region (defaults to us-east-1)
    """
    global _opensearch_client

    # Return existing client if already initialized
    if _opensearch_client is not None:
        return _opensearch_client

    host = os.environ.get("OPENSEARCH_ENDPOINT")
    region = os.environ.get("AWS_REGION", "us-east-1")

    # Determine service type based on endpoint
    if ".aoss." in host:
        service = "aoss"  # Amazon OpenSearch Serverless
    else:
        service = "es"  # Amazon OpenSearch Service (managed clusters)
    
    # Use custom credentials if available
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID_CUSTOM')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY_CUSTOM')
    
    if aws_access_key_id and aws_secret_access_key:
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region
        )
    else:
        session = boto3.Session()
        
    credentials = session.get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)

    # Configure OpenSearch client based on service type
    if service == "aoss":
        # OpenSearch Serverless configuration
        _opensearch_client = OpenSearch(
            hosts=[{"host": host[8:], "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
            pool_maxsize=20,
        )
    else:
        # OpenSearch Service (managed clusters) configuration
        _opensearch_client = OpenSearch(
            hosts=[{"host": host.replace("https://", ""), "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
            pool_maxsize=20,
        )

    return _opensearch_client


def get_embedding(text):
    """
    Get embeddings for the given text using Cohere Multilingual model via AWS Bedrock

    Args:
        text (str): The text to generate embeddings for

    Returns:
        list: The embedding vector
    """
    # Use custom credentials if available
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID_CUSTOM')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY_CUSTOM')
    
    if aws_access_key_id and aws_secret_access_key:
        bedrock_client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name='us-east-1'
        )
    else:
        bedrock_client = boto3.client("bedrock-runtime")

    # Prepare the request body for Cohere Embed model
    request_body = {"texts": [text], "input_type": "search_query", "truncate": "END"}

    try:
        # Invoke the model
        response = bedrock_client.invoke_model(
            modelId="cohere.embed-multilingual-v3", body=json.dumps(request_body)
        )

        # Parse the response body
        response_body = json.loads(response["body"].read())

        # Extract the embedding vector
        embedding = response_body["embeddings"][0]

        return embedding
    
    except Exception as e:
        print(f"Error calling Cohere model: {e}")
        print("Falling back to mock embeddings...")
        
        # Fallback to mock embeddings if Cohere fails
        import hashlib
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        dimension = 1024  # Cohere multilingual-v3 dimension
        embeddings = []
        
        for i in range(dimension):
            byte_index = i % len(hash_bytes)
            normalized_value = (hash_bytes[byte_index] - 127.5) / 127.5
            embeddings.append(normalized_value)
        
        return embeddings


def find_eccn_info_by_eccn_code(eccn_code, size=5):
    """
    Query OpenSearch by ECCN code

    Args:
        eccn_code: ECCN code to search for
        size: Maximum number of results to return

    Returns:
        dict: Search results
    """

    if eccn_code == "EAR99":

        return {
            "total_hits": 1,
            "results": [
                {
                    "eccn_code": "EAR99",
                    "description": """ Most of the products, services, and technologies that fall within the scope of the Export Administration Regulations (EAR) are not specifically controlled for export, and are given the classification of EAR99. They fall under U.S. Department of Commerce jurisdiction and are not listed on the Commerce Control List (CCL). EAR99 items generally consist of low-technology consumer goods and do not require a license in most situations.  
EAR99 items can generally be exported without a license but exporters of EAR99 items still need to perform careful due diligence to ensure the item is not going to an embargoed or sanctioned country, a prohibited end-user, or used in a prohibited end-use. If your proposed export of an EAR99 item is to an embargoed country, to an end-user of concern, or in support of a prohibited end-use, you may be required to obtain a license. 
An ECCN (short for Export Control Classification Number), is a designation that an item, which can be a tangible or intangible (i.e., software or technology), is controlled because of its specific performance characteristics, qualities, or designed-end use. Unlike an EAR99 designation, which is a broad basket category, an ECCN is much more narrowly defined and are focused on product categories. An ECCN is a five-digit alphanumeric designations that categorize items based on the nature of the product, i.e. type of commodity, software, or technology and its respective technical parameters. An example of an ECCN is 0A979, which corresponds to police helmets and shields. 
An ECCN is enumerated on the CCL, and each one lists important information that includes a general description of the controlled item(s), the reason(s) for control, available license exceptions, and, when necessary, additional details on related controls, and more specific item definitions. Learn more by viewing the link on how to determine your ECCN.                 
                """,
                }
            ],
        }
    client = get_opensearch_client()
    index_name = os.environ.get("OPENSEARCH_INDEX")

    try:
        # Build the query - using term query for exact matching on keyword field
        query = {
            "size": size,
            "query": {"term": {"eccn_code": eccn_code}},
            "_source": ECCN_SOURCES,
        }

        # Execute the query
        response = client.search(body=query, index=index_name)

        # Process results for better readability
        hits = response.get("hits", {}).get("hits", [])

        results = [format_search_result(hit.get("_source", {}), hit) for hit in hits]

        return {
            "total_hits": response.get("hits", {}).get("total", {}).get("value", 0),
            "results": results,
        }

    except Exception as e:
        raise Exception(f"Failed to query OpenSearch: {str(e)}")


def semantic_search(query_text, size=5, eccn_category=None):
    """
    Perform semantic search using Cohere embeddings via AWS Bedrock
    with optional filtering by ECCN category

    Args:
        query_text: Text to search for
        size: Maximum number of results to return
        eccn_category: Optional ECCN category to filter results

    Returns:
        dict: Search results
    """
    client = get_opensearch_client()
    index_name = os.environ.get("OPENSEARCH_INDEX")

    try:
        # Generate embeddings for the query
        embedding_vector = get_embedding(query_text)

        # Build the query based on whether we have a category filter
        if eccn_category:
            # Combined query with KNN search and category filter
            query = {
                "size": size,
                "query": {
                    "knn": {
                        "content_embedding": {
                            "vector": embedding_vector,
                            "k": size * 10,
                            "filter": {"term": {"eccn_category": eccn_category}},
                        }
                    }
                },
                "_source": ECCN_SOURCES,
            }
            # query = {
            #     "size": size,  # This ensures we only get 'size' results back
            #     "query": {
            #         "bool": {
            #             "must": [
            #                 {
            #                     "knn": {
            #                         "content_embedding": {
            #                             "vector": embedding_vector,
            #                             "k": size
            #                             * 50,  # Increased to ensure enough candidates after filtering
            #                         }
            #                     }
            #                 }
            #             ],
            #             "filter": [{"term": {"eccn_category": eccn_category}}],
            #         }
            #     },
            #     "_source": ECCN_SOURCES,
            # }
        else:
            # Standard KNN query without filtering
            query = {
                "size": size,
                "query": {
                    "knn": {
                        "content_embedding": {"vector": embedding_vector, "k": size}
                    }
                },
                "_source": ECCN_SOURCES,
            }

        # Execute the query
        response = client.search(body=query, index=index_name)

        # Process results for better readability
        hits = response.get("hits", {}).get("hits", [])

        # Format each search result
        results = [format_search_result(hit.get("_source", {}), hit) for hit in hits]

        return {
            "total_hits": response.get("hits", {}).get("total", {}).get("value", 0),
            "results": results,
        }

    except Exception as e:

        raise Exception(f"Failed to perform semantic search: {str(e)}")


def get_unique_categories():
    """
    Get all unique ECCN categories from the OpenSearch index

    Returns:
        dict: List of unique ECCN categories
    """
    client = get_opensearch_client()
    index_name = os.environ.get(
        "OPENSEARCH_INDEX", "eccn"
    )  # Default to "eccn" if not set

    try:

        query = {
            "size": 0,  # We don't need document results, just aggregations
            "aggs": {
                "unique_categories": {
                    "terms": {
                        "field": "eccn_category",
                        "size": 20,  # Get up to 20 unique categories
                    },
                    "aggs": {
                        "top_category_hits": {
                            "top_hits": {
                                "size": 1,
                                "_source": ["eccn_category", "eccn_category_detail"],
                            }
                        }
                    },
                }
            },
        }

        # Execute the query
        response = client.search(body=query, index=index_name)

        # Extract the unique categories with their details
        categories_with_details = []
        buckets = (
            response.get("aggregations", {})
            .get("unique_categories", {})
            .get("buckets", [])
        )

        for bucket in buckets:
            category = bucket.get("key")
            # Get the first hit that contains the category detail
            top_hit = (
                bucket.get("top_category_hits", {}).get("hits", {}).get("hits", [])
            )

            if top_hit:
                category_detail = (
                    top_hit[0].get("_source", {}).get("eccn_category_detail", "")
                )
                categories_with_details.append(
                    {"eccn_category": category, "eccn_category_detail": category_detail}
                )
            else:
                categories_with_details.append(
                    {"eccn_category": category, "eccn_category_detail": ""}
                )

        return {
            "total_categories": len(categories_with_details),
            "category_list": categories_with_details,
        }

    except Exception as e:
        print(f"Error in get_unique_categories: {str(e)}")
        raise Exception(f"Failed to get unique categories: {str(e)}")


def get_product_groups_by_category(eccn_category, min_doc_count=1, size=1000):
    """
    Retrieve unique product groups with their details from the OpenSearch index.

    Args:
        eccn_category: ECCN category filter (can be exact first digit for eccn code)
        min_doc_count: Minimum document count for a product group to be returned
        size: Maximum number of product groups to return

    Returns:
        Dictionary with product groups and their details
    """
    client = get_opensearch_client()
    index_name = os.environ.get("OPENSEARCH_INDEX", "eccn")

    # Build the aggregation query with required category filter
    query = {
        "size": 0,  # We don't need document results, just aggregations
        "query": {"bool": {"must": [{"term": {"eccn_category": eccn_category}}]}},
        "aggs": {
            "unique_product_groups": {
                "terms": {
                    "field": "product_group",
                    "size": size,
                    "min_doc_count": min_doc_count,
                    "order": {
                        "_count": "desc"
                    },  # Order by document count (most frequent first)
                },
                "aggs": {
                    "top_group_hits": {
                        "top_hits": {
                            "size": 1,
                            "_source": ["product_group", "product_group_detail"],
                        }
                    }
                },
            }
        },
    }

    try:
        # Execute the query
        response = client.search(index=index_name, body=query)

        # Extract the unique product groups with their details
        buckets = (
            response.get("aggregations", {})
            .get("unique_product_groups", {})
            .get("buckets", [])
        )

        # Get more detailed information from buckets
        product_group_data = []
        for bucket in buckets:
            group_key = bucket.get("key")
            doc_count = bucket.get("doc_count")

            # Get the product_group_detail from the top hit
            top_hit = bucket.get("top_group_hits", {}).get("hits", {}).get("hits", [])
            group_detail = ""
            if top_hit:
                group_detail = (
                    top_hit[0].get("_source", {}).get("product_group_detail", "")
                )

            product_group_data.append(
                {
                    "group_product": group_key,
                    "group_product_detail": group_detail,
                }
            )

        # Return structured response
        return {
            "total_group": len(product_group_data),
            "group_name_list": product_group_data,
        }

    except Exception as e:
        print(f"Error retrieving product groups: {e}")
        return {
            "error": str(e),
            "total_groups": 0,
            "product_groups": [],
            "simple_list": [],
            "group_details": {},
        }


def find_eccn_details_with_category_production_group(
    eccn_category,
    product_group,
    size=10,
):
    """
    Search with multiple filter options for ECCN category and product group filtering

    Args:
        eccn_category: ECCN category to filter results (e.g., '5', '3')
        product_group: product group to filter results
        size: Maximum number of results to return

    Returns:
        dict: Search results
    """
    client = get_opensearch_client()
    index_name = os.environ.get("OPENSEARCH_INDEX")

    try:
        # Build the query based on provided filters
        bool_query = {"bool": {"filter": []}}

        # Add category filter if provided
        if eccn_category:
            bool_query["bool"]["filter"].append(
                {"prefix": {"eccn_category": eccn_category}}
            )

        # Add product group filter if provided
        if product_group:
            bool_query["bool"]["filter"].append(
                {"match": {"product_group": product_group}}
            )

        # If no filters were provided, return an empty result
        if not eccn_category and not product_group:
            return {
                "total_hits": 0,
                "results": [],
                "message": "At least one filter parameter must be provided",
            }

        # Construct the final query
        query = {
            "size": size,
            "query": bool_query,
            "_source": ECCN_SOURCES,
        }

        # Execute the query
        response = client.search(body=query, index=index_name)

        # Process results for better readability
        hits = response.get("hits", {}).get("hits", [])

        results = [format_search_result(hit.get("_source", {}), hit) for hit in hits]

        return {
            "total_hits": response.get("hits", {}).get("total", {}).get("value", 0),
            "results": results,
        }

    except Exception as e:
        raise Exception(f"Failed to perform advanced search: {str(e)}")


def get_tools_definition():
    return [
        {
            "toolSpec": {
                "name": "semantic_search",
                "description": """Search for ECCN records using semantic similarity matching against official ECCN descriptions.
    
    This function should be used when you have a technical description, capability statement,
    or characteristic of a product, and need to find the most relevant ECCN classification.
    The search uses semantic similarity to match your query against official ECCN descriptions
    as defined by US export control regulations.
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
                "description": "Look up an ECCN record by the five digit eccn code. Use this when the query contains a specific ECCN code like '5A002' or '3A001'.",
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
                "description": "Get all unique ECCN categories from the database.",
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
                "description": "Get all unique product groups for a given ECCN category.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "eccn_category": {
                                "type": "string",
                                "description": "Optional ECCN category to filter results (e.g., 'A', 'B'). If not provided, returns all unique product groups.",
                            }
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "find_eccn_details_with_category_production_group",
                "description": "Search with filtering options for ECCN category and product group.",
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
                        "required": [],
                    }
                },
            }
        },
    ]


# Example usage (commented out)
# print(get_unique_categories())
# print(get_product_groups_by_category("1"))
# print(find_eccn_info_by_eccn_code("0A501"))
# print(
#     semantic_search(
#         "signal converter telecommunications RS-422 TTL interface hardware component low data rate",
#     )
# )

# print(
#     semantic_search(
#         "signal converter telecommunications RS-422 TTL interface hardware component low data rate",
#         eccn_category="5",
#     )
# )
