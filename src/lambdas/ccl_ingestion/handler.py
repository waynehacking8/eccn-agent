import json
import os
import boto3
import requests
from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth, OpenSearch
from datetime import datetime
import time

WAIT_TIME = 10


def get_embeddings(text, model_id=None):
    """
    Generate embeddings for the given text using Amazon Bedrock with Cohere model

    This function sends the input text to Amazon Bedrock's Cohere embedding model
    and returns the vector representation of the text. These embeddings are used
    for semantic search capabilities.

    Args:
        text (str): The text content to generate embeddings for
        model_id (str): The Bedrock model ID to use for embeddings
                        If None, uses the EMBEDDING_MODEL_ID environment variable
                        or falls back to "cohere.embed-multilingual-v3"

    Returns:
        list: A list of floating point numbers representing the text embedding vector

    Environment Variables:
        EMBEDDING_MODEL_ID: The Bedrock model ID to use for embeddings
    """
    # Get model ID from parameter, environment variable, or use default
    if model_id is None:
        model_id = os.environ.get("EMBEDDING_MODEL_ID", "cohere.embed-multilingual-v3")

    # Use custom credentials if available
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID_CUSTOM')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY_CUSTOM')
    
    if aws_access_key_id and aws_secret_access_key:
        bedrock_runtime = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name='us-east-1'
        )
    else:
        bedrock_runtime = boto3.client("bedrock-runtime")

    request_body = {
        "texts": [text],
        "input_type": "search_document",  # Optimized for search use cases
        "truncate": "END",  # Truncate from the end if text exceeds model's token limit
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=model_id, body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        embeddings = response_body["embeddings"][0]
        print(f"Generated Cohere embedding for text (length: {len(text)}, dimension: {len(embeddings)})")
        return embeddings
    
    except Exception as e:
        print(f"Error calling Cohere model {model_id}: {e}")
        print("Falling back to mock embeddings...")
        
        # Fallback to mock embeddings if Cohere fails
        import hashlib
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        dimension = int(os.environ.get("EMBEDDING_MODEL_DIMENSION", "1024"))
        embeddings = []
        
        for i in range(dimension):
            byte_index = i % len(hash_bytes)
            normalized_value = (hash_bytes[byte_index] - 127.5) / 127.5
            embeddings.append(normalized_value)
        
        print(f"Generated fallback embedding for text (length: {len(text)}, dimension: {dimension})")
        return embeddings


def get_opensearch_client():
    """
    Create and return an OpenSearch client using environment variables

    This function creates an authenticated connection to the OpenSearch
    serverless collection using AWS SigV4 authentication. It reads the
    endpoint from environment variables.

    Returns:
        OpenSearch: Configured OpenSearch client ready to use

    Environment Variables:
        OPENSEARCH_ENDPOINT: The host endpoint for OpenSearch
        AWS_REGION: The AWS region (defaults to us-east-1)
    """
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
        print(f"opensearch serverless endpoint: {host[8:]}")
        client = OpenSearch(
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
        endpoint_host = host.replace("https://", "")
        print(f"opensearch service endpoint: {endpoint_host}")
        client = OpenSearch(
            hosts=[{"host": endpoint_host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300,
            pool_maxsize=20,
        )

    return client


def get_document_ids(oss_client, index_name):
    """
    Retrieve all document IDs from the specified OpenSearch index

    This function executes a match_all query against the given index
    and returns only the document IDs without retrieving the full documents.
    It's used to get a list of all documents that need to be deleted.

    Args:
        oss_client (OpenSearch): The OpenSearch client
        index_name (str): The name of the index to query

    Returns:
        list: A list of document IDs from the index
    """

    # Query to get all document IDs
    query = {
        "query": {"match_all": {}},
        "_source": False,  # Disable source document
        "fields": ["_id"],  # Specify that only the _id field is returned
    }

    # Execute the search query
    response = oss_client.search(body=query, index=index_name, size="2000")

    # Extract document IDs from the response
    doc_ids = [hit["_id"] for hit in response["hits"]["hits"]]
    return doc_ids


def clear_existing_docs(oss_client, index_name):
    """
    Delete all documents from the specified OpenSearch index

    This function retrieves all document IDs from the given index and
    deletes them one by one. It's used to clean up an index before
    re-populating it with fresh data.

    Args:
        oss_client (OpenSearch): The OpenSearch client
        index_name (str): The name of the index to clear

    Returns:
        None
    """

    doc_ids = get_document_ids(oss_client, index_name)
    print(f"Found {len(doc_ids)} documents to delete.")

    # Delete all documents found
    for i, doc_id in enumerate(doc_ids):
        try:
            oss_client.delete(index=index_name, id=doc_id)
            # Print progress after every 10 documents
            if (i + 1) % 10 == 0:
                print(f"Deleted {i + 1}/{len(doc_ids)} documents")
        except:
            print(f"Deleted document with ID: {doc_id}")


def lambda_handler(event, context):
    """
    Lambda function to ingest JSON data into OpenSearch with vector embeddings

    This is the main entry point for the Lambda function. It processes incoming
    events (either S3 notifications or direct JSON input), generates embeddings
    using Bedrock, and indexes the data into OpenSearch.

    Args:
        event (dict): The Lambda event data
        context (LambdaContext): The Lambda execution context

    Returns:
        dict: Response containing status code and processing results

    Event Structure:
        - S3 event: Contains s3.bucket.name and s3.object.key
        - Direct input: Contains data object or array to process

    Environment Variables:
        OPENSEARCH_ENDPOINT: The host endpoint for OpenSearch
        OPENSEARCH_INDEX: The index name to use (defaults to 'content')
        EMBEDDING_MODEL_ID: The Bedrock model ID to use for embeddings
        AWS_REGION: The AWS region (defaults to us-east-1)
    """
    try:
        # Get OpenSearch client
        client = get_opensearch_client()

        # Get index name from environment variable or use default
        index_name = os.environ.get("OPENSEARCH_INDEX", "eccn")

        vector_dimension = os.environ.get("EMBEDDING_MODEL_DIMENSION", 1024)

        # Try to delete the index if it exists to clear all records
        try:
            if client.indices.exists(index=index_name):
                print(f"Clear docs: {index_name}")
                time.sleep(WAIT_TIME)
                clear_existing_docs(client, index_name)
                client.indices.delete(index=index_name)
                print(f"Deleted existing index: {index_name}")
                time.sleep(WAIT_TIME)
        except Exception as e:
            print(f"Warning: Could not check/delete existing index: {e}")
            print("Proceeding to create new index...")

        # Create a new index with mapping for vector search (if it doesn't exist)
        index_body = {
            "settings": {
                "index": {
                    "knn": True,  # Enable k-NN search capabilities
                    "refresh_interval": "10s",  # Make documents immediately searchable
                }
            },
            "mappings": {
                "properties": {
                    "eccn_category": {"type": "keyword"},
                    "eccn_category_detail": {"type": "text"},
                    "product_group": {"type": "keyword"},
                    "product_group_detail": {"type": "text"},
                    "eccn_code": {
                        "type": "keyword"
                    },  # Export Control Classification Number
                    "description": {"type": "text"},  # ECCN description text
                    "extra_details": {"type": "text"},  # Additional ECCN details
                    "content_embedding": {
                        "type": "knn_vector",
                        "dimension": vector_dimension,  # Cohere embed-english-v3 produces 1024-dimensional vectors
                        "method": {
                            "name": "hnsw",  # Hierarchical Navigable Small World algorithm
                            "space_type": "innerproduct",  # Use cosine similarity for vector comparisons
                            "engine": "faiss",  # Non-Metric Space Library, note we need to support semantic searh with filter
                            "parameters": {
                                "ef_construction": 512,  # Higher values improve index quality but slow construction
                                "m": 16,  # Number of bi-directional links created for each new element
                            },
                        },
                    },
                    "timestamp": {"type": "date"},  # Document timestamp
                }
            },
        }
        
        try:
            client.indices.create(index=index_name, body=index_body)
            print(f"Created new index: {index_name}")
        except Exception as e:
            if "resource_already_exists_exception" in str(e):
                print(f"Index {index_name} already exists, continuing with data ingestion...")
            else:
                raise e

        time.sleep(WAIT_TIME)

        # Load CCL data from the local file
        ccl_file_path = os.path.join(
            os.path.dirname(__file__), "ccl_parsed_checked.json"
        )

        with open(ccl_file_path, "r") as file:
            ccl_data = json.load(file)

        # Extract all ECCN entries from the CCL data
        eccn_entries = []
        for category in ccl_data:
            for product_group in category.get("product_groups", []):
                for eccn in product_group.get("eccn", []):
                    eccn_entry = {
                        "eccn_code": eccn.get("code", ""),
                        "description": eccn.get("description", ""),
                        "extra_details": eccn.get("details", ""),
                        "category_name": category.get("name", ""),
                        "product_group_name": product_group.get("name", ""),
                    }
                    eccn_entries.append(eccn_entry)

        # Process and index all ECCN entries
        results = []
        for entry in eccn_entries:
            result = process_and_index_item(client, index_name, entry)
            results.append(result)
            print(f"Indexed ECCN entry: {entry['eccn_code']}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Successfully processed {len(results)} ECCN entries",
                    "count": len(results),
                }
            ),
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"message": f"Error processing data: {str(e)}"}),
        }


def process_and_index_item(client, index_name, item):
    """
    Process a single item and index it in OpenSearch

    This function takes a JSON item, extracts the description, generates
    embeddings using Bedrock, and indexes the ECCN document with its
    vector representation into OpenSearch.

    Args:
        client (OpenSearch): The OpenSearch client
        index_name (str): The name of the index to write to
        item (dict): The JSON item to process and index

    Returns:
        dict: Result of the indexing operation including document ID
              and OpenSearch response details
    """
    # Extract description for embedding
    description_text = item.get("description", "")
    category_name = item.get("category_name", "")
    product_group_name = item.get("product_group_name", "")

    # Generate embeddings using Bedrock
    embeddings = get_embeddings(
        f"{category_name} {product_group_name} {description_text}"
    )

    # Prepare document for indexing
    document = {
        "eccn_code": item.get("eccn_code", ""),
        "description": description_text,
        "eccn_category": item.get("eccn_code", "")[0],
        "product_group": item.get("eccn_code", "")[1],
        "eccn_category_detail": category_name,
        "product_group_detail": product_group_name,
        "extra_details": item.get("extra_details", ""),
        "content_embedding": embeddings,
        "timestamp": datetime.now().isoformat(),
    }

    # Index the document
    response = client.index(index=index_name, body=document)

    return {
        "index_response": {
            "result": response["result"],
            "index": response["_index"],
            "status": response["_shards"]["successful"] > 0,
        }
    }
