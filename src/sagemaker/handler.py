import importlib
import json
import os
import re
import tempfile
from time import sleep
import base64
from urllib.parse import unquote_plus

import boto3

try:
    import pymupdf4llm
    PDF_PROCESSING_AVAILABLE = True
except ImportError:
    PDF_PROCESSING_AVAILABLE = False
    print("Warning: pymupdf4llm not available, PDF processing will be disabled")

from prompt import SYSTEM_PROMPT
from tools import (
    get_tools_definition,
)

import logging

logging.basicConfig(level=logging.WARNING)

# Configure logging based on environment variables using default format.
# Get log level from environment variable, default to INFO if not set
log_level_str = os.environ.get("LOG_LEVEL", "ERROR").upper()

# Map string log levels to logging constants
log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Get the numeric log level, default to INFO if invalid level specified
log_level = log_level_map.get(log_level_str, logging.INFO)

logger = logging.getLogger("eccn_agent_fc")
logger.setLevel(log_level)


def parse_multipart_form_data(event):
    """
    Parse multipart/form-data from Lambda Function URL event
    Returns: dict with parsed form fields and files
    """
    try:
        print(f"DEBUG: Full event structure: {json.dumps(event, default=str, indent=2)}")
        
        # Handle both API Gateway and Function URL formats
        headers = event.get('headers', {})
        
        # Function URL uses lowercase headers
        content_type = headers.get('content-type') or headers.get('Content-Type', '')
        
        print(f"DEBUG: Content-Type: {content_type}")
        print(f"DEBUG: Event keys: {list(event.keys())}")
        
        if not content_type or not content_type.startswith('multipart/form-data'):
            print(f"DEBUG: Not multipart data: {content_type}")
            return None
            
        # Extract boundary from content-type header
        boundary_match = re.search(r'boundary=([^;]+)', content_type)
        if not boundary_match:
            print(f"DEBUG: No boundary found in content-type: {content_type}")
            return None
            
        boundary = boundary_match.group(1).strip('"')
        body = event.get('body', '')
        
        print(f"DEBUG: Boundary: {boundary}")
        print(f"DEBUG: Body type: {type(body)}, Length: {len(body) if body else 0}")
        print(f"DEBUG: isBase64Encoded: {event.get('isBase64Encoded', False)}")
        
        if not body:
            print("DEBUG: Empty body")
            return None
        
        # Decode if base64 encoded
        if event.get('isBase64Encoded', False):
            body_bytes = base64.b64decode(body)
        else:
            body_bytes = body.encode('utf-8') if isinstance(body, str) else body
        
        # Split by boundary
        boundary_bytes = f'--{boundary}'.encode('utf-8')
        parts = body_bytes.split(boundary_bytes)
        
        parsed_data = {
            'fields': {},
            'files': []
        }
        
        for part in parts[1:-1]:  # Skip first empty and last closing parts
            if not part.strip():
                continue
                
            # Convert part to string for header parsing, but preserve binary content
            try:
                part_str = part.decode('utf-8', errors='ignore')
            except:
                continue
                
            # Split headers and content - fix CRLF handling
            header_end = part_str.find('\r\n\r\n')
            if header_end == -1:
                header_end = part_str.find('\n\n')
                header_separator = b'\n\n'
                header_separator_len = 2
            else:
                header_separator = b'\r\n\r\n'
                header_separator_len = 4
                
            if header_end == -1:
                print(f"DEBUG: No header separator found in part")
                continue
            
            headers = part_str[:header_end]
            print(f"DEBUG: Headers found: {headers}")
            
            # For binary content, we need to work with the original bytes
            header_end_bytes = part.find(header_separator)
            if header_end_bytes == -1:
                print(f"DEBUG: No header separator found in bytes")
                continue
                
            header_end_bytes += header_separator_len
            content_bytes = part[header_end_bytes:].rstrip(b'\r\n')
            
            # Parse Content-Disposition header
            content_disposition = ''
            for header_line in headers.split('\n'):
                if header_line.lower().startswith('content-disposition:'):
                    content_disposition = header_line.strip()
                    break
            
            if not content_disposition:
                continue
            
            # Extract name and filename
            print(f"DEBUG: Content-Disposition: {content_disposition}")
            name_match = re.search(r'name="([^"]*)"', content_disposition)
            filename_match = re.search(r'filename="([^"]*)"', content_disposition)
            
            if not name_match:
                print(f"DEBUG: No name found in Content-Disposition")
                continue
                
            field_name = name_match.group(1)
            filename = filename_match.group(1) if filename_match else None
            print(f"DEBUG: Field name: {field_name}, Filename: {filename}")
            
            if filename:  # This is a file
                parsed_data['files'].append({
                    'name': field_name,
                    'filename': filename,
                    'content': content_bytes
                })
            else:  # This is a regular form field
                try:
                    field_value = content_bytes.decode('utf-8')
                    parsed_data['fields'][field_name] = field_value
                except:
                    # If decoding fails, store as bytes
                    parsed_data['fields'][field_name] = content_bytes
        
        return parsed_data
        
    except Exception as e:
        logger.error(f"Error parsing multipart form data: {e}")
        return None


def process_uploaded_files(files):
    """
    Process uploaded files from multipart form data
    Returns: (product_description_text, processed_pdf_files)
    """
    product_description = ""
    processed_files = []
    
    for file_info in files:
        filename = file_info['filename'].lower()
        content = file_info['content']
        
        if filename.endswith('.txt'):
            # Extract text content for product description
            try:
                text_content = content.decode('utf-8')
                product_description += text_content + "\\n"
            except UnicodeDecodeError:
                logger.warning(f"Could not decode text file {filename}")
                
        elif filename.endswith('.pdf') and PDF_PROCESSING_AVAILABLE:
            # Process PDF file
            try:
                # Write PDF content to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                # Convert PDF to markdown using pymupdf4llm
                md_text = pymupdf4llm.to_markdown(temp_file_path)
                
                # Clean up temp file
                os.unlink(temp_file_path)
                
                # Add to processed files
                processed_files.append({
                    "name": filename[:-4],  # Remove .pdf extension
                    "content": md_text
                })
                
            except Exception as e:
                logger.warning(f"Error processing PDF {filename}: {e}")
        
        elif filename.endswith('.pdf') and not PDF_PROCESSING_AVAILABLE:
            logger.warning(f"PDF processing not available for {filename}")
    
    return product_description.strip(), processed_files


def build_user_message_from_direct_input(test_case: dict) -> list:
    """
    Build user message from direct input test case
    
    Args:
        test_case (dict): Dictionary containing:
            - product_description: Text describing the product
            - pdf_files: List of PDF file objects with name and content
            
    Returns:
        list: List of message parts for Claude
    """
    result = []
    
    # Add product description - this is required
    if "product_description" not in test_case or not test_case["product_description"]:
        raise ValueError("Test case is missing required product_description")
    
    result.append({"text": test_case["product_description"]})
    
    # Add PDF documents if available
    if "pdf_files" in test_case and test_case["pdf_files"]:
        for pdf_file in test_case["pdf_files"]:
            if "name" in pdf_file and "content" in pdf_file:
                result.append({
                    "document": {
                        "format": "txt",
                        "name": pdf_file["name"],
                        "source": {"bytes": pdf_file["content"].encode("utf-8")},
                    }
                })
    
    return result


def filter_fields(obj, field_to_exclude="bytes"):
    """Filter function to exclude any fields with the specified name"""
    if isinstance(obj, dict):
        return {
            k: filter_fields(v, field_to_exclude)
            for k, v in obj.items()
            if k != field_to_exclude
        }
    elif isinstance(obj, list):
        return [filter_fields(item, field_to_exclude) for item in obj]
    else:
        return obj


def llm_with_tool_reasoning(
    bedrock_client,
    messages,
    tools,
    model_id,
    max_tokens=8000,
    temperature=0.2,
    system_prompt=SYSTEM_PROMPT,
):
    """
    Invoke Claude 3.7 Sonnet with tool use and reasoning using the converse API

    Args:
        bedrock_client: Boto3 client for Bedrock
        messages: Complete conversation history
        tools: List of tool definitions
        model_id: The Claude model ID to use
        max_tokens: Maximum tokens for response
        temperature: Temperature for response generation

    Returns:
        tuple: (stop_reason, assistant_response, has_tool_calls, tool_use_blocks)
    """
    # Construct the tool configuration
    tool_config = {"tools": tools}

    # Disable thinking mode for stability
    reasoning_config = {}

    # Log model input
    logger.debug(
        f"Model Input - Messages: {json.dumps(filter_fields(messages), default=str)}"
    )

    # Invoke Claude using converse API
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=[{"text": system_prompt}],
            toolConfig=tool_config,
            additionalModelRequestFields=reasoning_config,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )

        # Extract the message from the response
        assistant_message = response.get("output", {}).get("message", {})

        # Filter out reasoning content to avoid SDK_UNKNOWN_MEMBER errors
        if "content" in assistant_message:
            filtered_content = []
            for content_block in assistant_message["content"]:
                if "SDK_UNKNOWN_MEMBER" not in str(content_block):
                    filtered_content.append(content_block)
            assistant_message["content"] = filtered_content

        # Log model output
        logger.info(
            f"Model assistant_message - Response: {json.dumps(assistant_message, default=str)}"
        )

        # Check for tool use
        has_tool_calls = False
        tool_use_blocks = []

        # Check if the stop reason is tool_use
        logger.info(f"Stop reason {response.get('stopReason')}")
        if response.get("stopReason") == "tool_use":
            has_tool_calls = True

            # Extract tool use blocks from the content
            for content_block in assistant_message["content"]:
                if "toolUse" in content_block.keys():
                    tool_use_blocks.append(content_block["toolUse"])
            logger.info(
                f"output from tool use: {json.dumps(tool_use_blocks, default=str)}"
            )
        
        return (
            response.get("stopReason"),
            assistant_message,
            has_tool_calls,
            tool_use_blocks,
        )

    except Exception as e:
        logger.error(f"Error invoking model with converse API: {str(e)}")
        raise Exception(f"Failed to invoke Claude with converse API: {str(e)}")


def process_tool_calls(tool_use_blocks):
    """
    Process the tool calls and generate tool results

    Args:
        tool_use_blocks: List of tool use blocks from Claude

    Returns:
        list: Tool results in the format required by Claude
    """
    tool_results = []

    # Define a mapping of tool names to their corresponding functions
    # Get the current module to look up functions
    tool_module_name = "tools"  # Replace with actual module path
    tool_module = importlib.import_module(tool_module_name)

    for tool_block in tool_use_blocks:
        tool_name = tool_block.get("name")
        tool_input = tool_block.get("input", {})
        tool_id = tool_block.get("toolUseId")

        try:
            if hasattr(tool_module, tool_name):
                logger.info(f"Calling tool: {tool_name}")
                logger.info(
                    f"Function call parameters {json.dumps(tool_input, indent=2)}"
                )
                function = getattr(tool_module, tool_name)

                tool_result = function(**tool_input)
                logger.info(
                    f"Function call result (extra_details field hidden): {json.dumps(filter_fields(tool_result, 'extra_details'), indent=2)}"
                )

            else:
                tool_result = {"error": f"Unknown tool: {tool_name}"}

            # Add the tool result to our results list with the correct format for converse API
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_id,
                        "content": [{"json": tool_result}],
                    }
                }
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            # Add error result with correct format for converse API
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_id,
                        "content": [
                            {"json": {"error": f"Error processing tool: {str(e)}"}}
                        ],
                    }
                }
            )

    return tool_results


def extract_final_text_from_response(response):
    """
    Extract the text content from Claude's final response
    """
    text_content = []

    for block in response.get("content", []):
        if "text" in block:
            text_content.append(block.get("text", ""))

    return " ".join(text_content)


def lambda_handler(event, context):
    """
    AWS Lambda handler that controls LLM calls with tool reasoning using a loop

    Expected event structure - supports multiple input formats:
    
    Format 1 - Multipart form data (for Postman file uploads):
    Content-Type: multipart/form-data
    Form fields:
    - product_description: text field (optional if .txt file is uploaded)
    - files: multiple file uploads (.txt for description, .pdf for technical docs)
    - size: text field (optional, defaults to 5)
    - bedrock_model_id: text field (optional)
    - max_tokens: text field (optional, defaults to 8000)
    - temperature: text field (optional, defaults to 1)
    - max_tool_iterations: text field (optional, defaults to 5)
    
    Format 2 - Direct text input (JSON format):
    {
        "test_case": {
            "product_description": "Product description text",
            "pdf_s3_locations": ["s3://bucket/path/file1.pdf", "s3://bucket/path/file2.pdf"],  # optional
            "pdf_files": [  # optional - for direct PDF upload
                {
                    "name": "datasheet.pdf",
                    "content": "base64_encoded_pdf_content"
                }
            ]
        },
        "size": 5,  # optional, defaults to 5
        "bedrock_model_id": "anthropic.claude-3-sonnet-20240229-v1",  # optional
        "use_thinking": true,  # optional, whether to use extended thinking
        "thinking_budget_tokens": 4000,  # optional, budget for thinking when enabled
        "max_tokens": 8000,  # optional, default response size
        "temperature": 0.2,  # optional, controls randomness
        "max_tool_iterations": 3  # optional, maximum number of tool usage iterations
    }
    
    Format 3 - S3 location input (original format):
    {
        "s3_location": "s3://bucket/prefix",  # location of project files
        "size": 5,  # optional, defaults to 5
        "bedrock_model_id": "anthropic.claude-3-sonnet-20240229-v1",  # optional
        "use_thinking": true,  # optional, whether to use extended thinking
        "thinking_budget_tokens": 4000,  # optional, budget for thinking when enabled
        "max_tokens": 8000,  # optional, default response size
        "temperature": 0.2,  # optional, controls randomness
        "max_tool_iterations": 3  # optional, maximum number of tool usage iterations
    }
    """
    try:
        # First check for multipart form data
        multipart_data = parse_multipart_form_data(event)
        
        if multipart_data:
            # Format 1: Multipart form data (for Postman file uploads)
            fields = multipart_data.get('fields', {})
            files = multipart_data.get('files', [])
            
            # Process uploaded files
            file_description, processed_files = process_uploaded_files(files)
            
            # Get product description from form field or uploaded .txt file
            form_description = fields.get('product_description', '').strip()
            product_description = form_description if form_description else file_description
            
            if not product_description:
                raise ValueError("product_description field or .txt file is required")
            
            # Create test_case format for consistency
            test_case = {
                "product_description": product_description,
                "pdf_files": processed_files
            }
            
            # Build user message from direct input
            original_message = build_user_message_from_direct_input(test_case)
            
            # Extract configuration from form fields
            max_tool_iterations = int(fields.get("max_tool_iterations", "5"))
            bedrock_model_id = fields.get("bedrock_model_id", 
                                        os.environ.get("DEFAULT_BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"))
            max_tokens = int(fields.get("max_tokens", "8000"))
            temperature = float(fields.get("temperature", "1"))
            
        elif "test_case" in event:
            # Format 2: Direct text input with optional S3 PDF references
            test_case = event.get("test_case", {})
            if not test_case.get("product_description"):
                raise ValueError("test_case.product_description is required")
            
            # Build user message from direct input
            original_message = build_user_message_from_direct_input(test_case)
            
            # Extract configuration from event
            max_tool_iterations = event.get("max_tool_iterations", 5)
            bedrock_model_id = event.get(
                "bedrock_model_id", 
                os.environ.get("DEFAULT_BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
            )
            max_tokens = event.get("max_tokens", 8000)
            temperature = event.get("temperature", 1)
            
        elif "s3_location" in event:
            # Format 3: S3 location input (original format)
            s3_location = event.get("s3_location", "")
            if not s3_location:
                raise ValueError("s3_location is required when using S3 format")
            
            # Process S3 content and update user query with project description
            original_message = build_user_message_from_s3_content(s3_location)
            
            # Extract configuration from event
            max_tool_iterations = event.get("max_tool_iterations", 5)
            bedrock_model_id = event.get(
                "bedrock_model_id", 
                os.environ.get("DEFAULT_BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
            )
            max_tokens = event.get("max_tokens", 8000)
            temperature = event.get("temperature", 1)
            
        else:
            raise ValueError("Either multipart form data, 'test_case', or 's3_location' must be provided")

        # Thinking configuration if enabled
        thinking_config = None
        if event.get("use_thinking", False):
            thinking_config = {
                "type": "enabled",
                "budget_tokens": event.get("thinking_budget_tokens", 4000),
            }
            temperature = 1  # if the thinking is enabled, temperature must be 1

        # Initialize clients with custom credentials if provided
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

        # Define the tools for Claude
        tools = get_tools_definition()

        # Initialize conversation with user query
        messages = [{"role": "user", "content": original_message}]

        all_responses = []
        has_tool_calls = False
        tool_use_blocks = []
        stop_reason = None

        # Main loop for tool reasoning with fixed iterations
        for iteration in range(max_tool_iterations + 1):  # +1 for initial query

            # Invoke Claude
            stop_reason, assistant_response, has_tool_calls, tool_use_blocks = (
                llm_with_tool_reasoning(
                    bedrock_client,
                    messages,
                    tools,
                    bedrock_model_id,
                    max_tokens,
                    temperature,
                )
            )

            if stop_reason == "stop_sequence":
                break

            # Add assistant response to conversation history
            messages.append(assistant_response)
            all_responses.append(assistant_response)

            # If no tool calls or we've reached the max iterations, exit the loop
            if not has_tool_calls or iteration == max_tool_iterations:
                break

            # Process tool calls
            tool_results = process_tool_calls(tool_use_blocks)

            # Add tool results as user message - properly formatted for converse API
            messages.append({"role": "user", "content": tool_results})

        # Prepare final response
        final_response = {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "stop_reason": stop_reason,
                    "tool_iterations_used": iteration,
                    "max_tool_iterations": max_tool_iterations,
                    "conversation": all_responses,
                    "final_response": extract_final_text_from_response(
                        all_responses[-1]
                    ),
                    "has_additional_tool_calls": has_tool_calls,
                },
                default=str,
            ),
        }
        logger.debug(json.dumps(final_response, indent=2, ensure_ascii=False))

        return final_response

    except ValueError as ve:
        return {"statusCode": 400, "body": json.dumps({"error": str(ve)})}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


def build_user_message_from_s3_content(s3_location: str) -> list:
    """
    Process content from an S3 location and update the user query with project description.

    Args:
        s3_location (str): S3 URI in the format 's3://bucket/prefix'

    Returns:
        list: List of message parts for Claude
    """

    result = []

    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Parse S3 location to get bucket and prefix
    if not s3_location or not s3_location.startswith("s3://"):
        raise ValueError("You have not provide a correct s3 path")

    parts = s3_location[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    # Read project description - this is required
    try:
        description_key = f"{prefix}/description.txt" if prefix else "description.txt"
        response = s3_client.get_object(Bucket=bucket, Key=description_key)
        project_description = response["Body"].read().decode("utf-8")
        result.append({"text": project_description})
    except Exception as e:
        raise ValueError("can not read description.txt in the location")

    # Read image if available (optional)
    try:
        image_key = f"{prefix}/image.jpg" if prefix else "image.jpg"
        response = s3_client.get_object(Bucket=bucket, Key=image_key)

        result.append(
            {"image": {"format": "png", "source": {"bytes": response["Body"].read()}}}
        )
    except Exception as e:
        logger.warning(f"Warning: Could not read image.jpg: {str(e)}")

    # List and collect PDF documents (optional)
    try:
        # List objects with PDF extension
        list_prefix = f"{prefix}/" if prefix else ""
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=list_prefix)
        need_to_cache = False
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                # For now, just store the S3 URI to the PDF
                response = s3_client.get_object(Bucket=bucket, Key=key)
                # Create a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(key)[1]
                ) as temp_file:
                    temp_file.write(response["Body"].read())
                    temp_file_path = temp_file.name

                # Use the temporary file path with pymupdf4llm
                if PDF_PROCESSING_AVAILABLE:
                    md_text = pymupdf4llm.to_markdown(temp_file_path)
                else:
                    md_text = "PDF processing not available"

                # Clean up the temporary file when done
                os.unlink(temp_file_path)

                result.append(
                    {
                        "document": {
                            "format": "txt",
                            "name": key.split("/")[-1][:-4],
                            "source": {"bytes": md_text.encode("utf-8")},
                        }
                    }
                )
                need_to_cache = True
    except Exception as e:
        logger.warning(f"Warning: Error listing PDF documents")

    return result


def lambda_handler_local_test(event, context):
    """
    Local test version of lambda_handler that handles test cases with local file paths
    """
    try:
        # Extract test case and prompt template
        test_case = event.get("test_case", {})
        prompt_template = event.get("prompt_template", "")
        
        # Build user message from test case
        original_message = build_user_message_from_local_test_case(test_case)
        
        # Configuration
        max_tool_iterations = 5
        bedrock_model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        max_tokens = 8000
        temperature = 1
        use_thinking = True
        thinking_budget_tokens = 4000
        
        # Thinking configuration if enabled
        thinking_config = None
        if use_thinking:
            thinking_config = {
                "type": "enabled",
                "budget_tokens": thinking_budget_tokens,
            }
        
        # Initialize Bedrock client
        bedrock_client = boto3.client("bedrock-runtime")
        
        # Define the tools for Claude
        tools = get_tools_definition()
        
        # Initialize conversation with user query
        messages = [{"role": "user", "content": original_message}]
        
        all_responses = []
        has_tool_calls = False
        tool_use_blocks = []
        stop_reason = None
        
        # Main loop for tool reasoning with fixed iterations
        for iteration in range(max_tool_iterations + 1):  # +1 for initial query
            
            # Invoke Claude
            system_prompt = prompt_template if prompt_template else SYSTEM_PROMPT
            stop_reason, assistant_response, has_tool_calls, tool_use_blocks = (
                llm_with_tool_reasoning(
                    bedrock_client,
                    messages,
                    tools,
                    bedrock_model_id,
                    max_tokens,
                    temperature,
                    system_prompt
                )
            )
            
            if stop_reason == "stop_sequence":
                break
            
            # Add assistant response to conversation history
            messages.append(assistant_response)
            all_responses.append(assistant_response)
            
            # If no tool calls or we've reached the max iterations, exit the loop
            if not has_tool_calls or iteration == max_tool_iterations:
                break
            
            # Process tool calls
            tool_results = process_tool_calls(tool_use_blocks)
            
            # Add tool results as user message - properly formatted for converse API
            messages.append({"role": "user", "content": tool_results})
        
        # Extract final response
        final_response_text = extract_final_text_from_response(assistant_response)
        
        return json.dumps({"final_response": final_response_text})
        
    except Exception as e:
        logger.error(f"Error in lambda_handler_local_test: {str(e)}")
        return json.dumps({"error": str(e)})


def build_user_message_from_local_test_case(test_case):
    """
    Build user message from local test case with file paths
    """
    result = []
    
    # Add product description
    product_description = test_case.get("product_description", "")
    if product_description:
        result.append({"text": f"Product Description: {product_description}"})
    
    # Add PDF file if available
    pdf_path = test_case.get("product_pdf", "")
    if pdf_path and os.path.exists(pdf_path):
        try:
            if PDF_PROCESSING_AVAILABLE:
                md_text = pymupdf4llm.to_markdown(pdf_path)
                result.append({
                    "document": {
                        "format": "txt",
                        "name": os.path.basename(pdf_path).replace('.pdf', ''),
                        "source": {"bytes": md_text.encode("utf-8")},
                    }
                })
            else:
                result.append({"text": f"PDF file available at: {pdf_path} (PDF processing not available)"})
        except Exception as e:
            result.append({"text": f"Error processing PDF {pdf_path}: {str(e)}"})
    
    # Add image file if available
    image_path = test_case.get("product_image", "")
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                result.append({
                    "image": {
                        "format": "png" if image_path.lower().endswith('.png') else "jpeg",
                        "source": {"bytes": img_data}
                    }
                })
        except Exception as e:
            result.append({"text": f"Error processing image {image_path}: {str(e)}"})
    
    return result