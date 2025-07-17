import json
import os
import sys
import boto3
from prompt import SYSTEM_PROMPT
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set AWS credentials from environment variables
# Note: Replace with your actual AWS credentials
# os.environ['AWS_ACCESS_KEY_ID'] = 'YOUR_ACCESS_KEY_ID'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'YOUR_SECRET_ACCESS_KEY'
os.environ['AWS_REGION'] = 'us-east-1'
os.environ['OPENSEARCH_ENDPOINT'] = 'https://search-eccn-agent-main-domain-otccw4tfus2ki55ehlo3gt3hja.us-east-1.es.amazonaws.com'
os.environ['OPENSEARCH_INDEX'] = 'eccn'

def lambda_handler_local_test(event, context):
    """
    Local test version of the Lambda handler
    """
    try:
        # Extract the query from the event
        query = event.get('query', '')
        
        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Query parameter is required'})
            }
        
        # Initialize AWS clients
        bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ['AWS_REGION'])
        opensearch_client = boto3.client('opensearchserverless', region_name=os.environ['AWS_REGION'])
        
        # Process the query and generate response
        response = process_query(query, bedrock_client, opensearch_client)
        
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def process_query(query, bedrock_client, opensearch_client):
    """
    Process the query using Bedrock and OpenSearch
    """
    # Implementation would go here
    # This is a placeholder for the actual query processing logic
    return {
        'message': 'Query processed successfully',
        'query': query
    }