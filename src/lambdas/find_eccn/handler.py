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

import logging

logging.basicConfig(level=logging.WARNING)

# Configure logging based on environment variables using default format.
log_level_str = os.environ.get("LOG_LEVEL", "ERROR").upper()

log_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

log_level = log_level_map.get(log_level_str, logging.INFO)

logger = logging.getLogger("eccn_agent_local")
logger.setLevel(log_level)

# Load the optimized Round 3 system prompt
def load_optimized_system_prompt():
    """Load the optimized system prompt based on Round 3 results"""
    return """You are an expert in US Export Control Classification specialized in determining the correct Export Control Classification Number (ECCN) for products based on the Commerce Control List (CCL).

## Your Task
Analyze the provided product information and determine the correct US Export Control Classification Number (ECCN). Follow the structured approach below to ensure accurate classification.

## Analysis Framework
1. First, create a detailed summary of the product's technical specifications and capabilities
2. Identify whether the product primarily falls under:
   - Category 4 (Computers)
   - Category 5 Part 1 (Telecommunications)
   - Category 5 Part 2 (Information Security)
   - Other CCL categories
   - Or potentially EAR99 (not specifically enumerated on the CCL)

3. For networking/telecommunications equipment, evaluate against these common ECCNs in this order:
   - 5A002: Does it perform cryptographic functions or contain encryption for data confidentiality?
      * Look for features like: VPN support, SSL/TLS encryption for data transmission, IPsec, encryption protocols, secure communications
      * Key terms: encryption, cryptography, security protocols, secure transmission
      * IMPORTANT: Standard protocols like SNMP, Modbus/TCP, PROFINET alone do NOT qualify as cryptographic functions
   
   - 5A992.c: Does it have "information security" features but doesn't meet 5A002 criteria?
      * Look for: encryption for authentication purposes only, 802.1x authentication with encryption options, password protection with encryption
      * IMPORTANT: Not all authentication is encryption - basic password protection or MAC filtering alone doesn't qualify for 5A992.c
   
   - 5A001: Does it have specialized telecommunications capabilities?
      * Look for: jamming equipment, interception features, specialized radio functions
   
   - 5A991: Is it telecommunications equipment without encryption or security features?
      * For Ethernet switches, routers, media converters: These typically fall under 5A991.b.1 as telecommunications switching equipment
      * Basic network equipment without encryption is generally 5A991.b.1, not 5A991.g
      * Industrial temperature ranges alone do not change the classification from 5A991.b to 5A991.a
   
   - 4A994: Is it computer equipment with network capabilities?
      * Computing devices rather than pure networking equipment
   
   - EAR99: Does it not fit any specific ECCN category?

4. IMPORTANT: Security Feature vs. Standard Protocol Evaluation
   - Standard network protocols (SNMP v1/v2, HTTP, Modbus/TCP, PROFINET, EtherNet/IP) are NOT security features
   - Basic QoS, VLAN, and similar traffic management features are NOT security features
   - Simple password protection or MAC filtering alone does NOT qualify as encryption
   - Actual security features include:
      * Encryption for data confidentiality (data in transit)
      * Cryptographic authentication (beyond simple passwords)
      * Encrypted management protocols (HTTPS, SSH, SNMPv3 with privacy)
      * Features explicitly described as implementing encryption or cryptography

5. For each relevant category, explicitly state:
   - The regulatory definition/threshold from the CCL
   - Whether the product meets or fails to meet that threshold
   - Specific evidence from the product specifications supporting your determination

## Key Classification Guidelines
- 5A002 is for products with substantial cryptographic capabilities for data confidentiality
- 5A992.c is for products with security features using cryptography that don't meet 5A002 thresholds
- 5A991.b.1 is the correct classification for most basic telecommunications switching equipment (including Ethernet switches, routers, media converters)
- 5A991 is for telecommunications equipment WITHOUT security features using cryptography
- Always provide complete classification including all applicable subcategories (e.g., 5A991.b.1, not just 5A991)

## Response Format
Structure your analysis as follows:

1. Product Summary
   - Technical specifications
   - Primary functions and capabilities

2. Classification Analysis
   - Primary CCL category evaluation
   - Evaluation against each relevant ECCN (must include 5A002, 5A992.c, 5A001, 5A991, 4A994, and EAR99)
   - For each ECCN, provide the regulatory definition, determination (applies/does not apply), and justification

3. Final Determination
   - State the correct ECCN with its complete sub-category (e.g., 5A991.b.1)
   - Provide a concise justification for this classification
   - List key factors that led to selecting this ECCN over other potential classifications

Be thorough, precise, and ensure your classification is fully supported by the CCL regulations."""

SYSTEM_PROMPT = load_optimized_system_prompt()

# Import local tools (they will be in the Lambda layer)
def import_local_tools():
    """Import local embedding tools from the Lambda layer"""
    try:
        # Try to import from layer path first
        import sys
        layer_path = '/opt/python/local_embeddings'
        if layer_path not in sys.path:
            sys.path.insert(0, layer_path)
        
        from tools import get_tools_definition
        return True
    except Exception as e:
        logger.error(f"Failed to import local tools: {e}")
        # Fallback to basic tools definition
        return False

def get_tools_definition_fallback():
    """Fallback tools definition when local tools are not available"""
    return [
        {
            "toolSpec": {
                "name": "semantic_search",
                "description": "Search for ECCN records using semantic similarity matching against official ECCN descriptions.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "query_text": {
                                "type": "string",
                                "description": "The technical description or characteristics to match against ECCN definitions.",
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
        }
    ]

# Parse multipart form data (same as original)
def parse_multipart_form_data(event):
    """Parse multipart/form-data from Lambda Function URL event"""
    try:
        print(f"DEBUG: Full event structure: {json.dumps(event, default=str, indent=2)}")
        
        headers = event.get('headers', {})
        content_type = headers.get('content-type') or headers.get('Content-Type', '')
        
        print(f"DEBUG: Content-Type: {content_type}")
        
        if not content_type or not content_type.startswith('multipart/form-data'):
            print(f"DEBUG: Not multipart data: {content_type}")
            return None
            
        boundary_match = re.search(r'boundary=([^;]+)', content_type)
        if not boundary_match:
            print(f"DEBUG: No boundary found in content-type: {content_type}")
            return None
            
        boundary = boundary_match.group(1).strip('"')
        body = event.get('body', '')
        
        if not body:
            print("DEBUG: Empty body")
            return None
        
        if event.get('isBase64Encoded', False):
            body_bytes = base64.b64decode(body)
        else:
            body_bytes = body.encode('utf-8') if isinstance(body, str) else body
        
        boundary_bytes = f'--{boundary}'.encode('utf-8')
        parts = body_bytes.split(boundary_bytes)
        
        parsed_data = {
            'fields': {},
            'files': []
        }
        
        for part in parts[1:-1]:
            if not part.strip():
                continue
                
            try:
                part_str = part.decode('utf-8', errors='ignore')
            except:
                continue
                
            header_end = part_str.find('\r\n\r\n')
            if header_end == -1:
                header_end = part_str.find('\n\n')
                header_separator = b'\n\n'
                header_separator_len = 2
            else:
                header_separator = b'\r\n\r\n'
                header_separator_len = 4
                
            if header_end == -1:
                continue
            
            headers = part_str[:header_end]
            
            header_end_bytes = part.find(header_separator)
            if header_end_bytes == -1:
                continue
                
            header_end_bytes += header_separator_len
            content_bytes = part[header_end_bytes:].rstrip(b'\r\n')
            
            content_disposition = ''
            for header_line in headers.split('\n'):
                if header_line.lower().startswith('content-disposition:'):
                    content_disposition = header_line.strip()
                    break
            
            if not content_disposition:
                continue
            
            name_match = re.search(r'name="([^"]*)"', content_disposition)
            filename_match = re.search(r'filename="([^"]*)"', content_disposition)
            
            if not name_match:
                continue
                
            field_name = name_match.group(1)
            filename = filename_match.group(1) if filename_match else None
            
            if filename:
                # Only accept PDF files
                if filename.lower().endswith('.pdf'):
                    parsed_data['files'].append({
                        'name': field_name,
                        'filename': filename,
                        'content': content_bytes
                    })
            else:
                try:
                    field_value = content_bytes.decode('utf-8')
                    parsed_data['fields'][field_name] = field_value
                except:
                    parsed_data['fields'][field_name] = content_bytes
        
        return parsed_data
        
    except Exception as e:
        logger.error(f"Error parsing multipart form data: {e}")
        return None

def process_uploaded_files(files):
    """Process uploaded PDF files"""
    product_description = ""
    processed_files = []
    
    for file_info in files:
        filename = file_info['filename'].lower()
        content = file_info['content']
        
        if filename.endswith('.pdf') and PDF_PROCESSING_AVAILABLE:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(content)
                    temp_file_path = temp_file.name
                
                md_text = pymupdf4llm.to_markdown(temp_file_path)
                os.unlink(temp_file_path)
                
                processed_files.append({
                    "name": filename[:-4],
                    "content": md_text
                })
                
            except Exception as e:
                logger.warning(f"Error processing PDF {filename}: {e}")
        
        elif filename.endswith('.pdf') and not PDF_PROCESSING_AVAILABLE:
            logger.warning(f"PDF processing not available for {filename}")
    
    return product_description.strip(), processed_files

def build_user_message_from_direct_input(test_case: dict) -> list:
    """Build user message from direct input test case"""
    result = []
    
    if "product_description" not in test_case or not test_case["product_description"]:
        raise ValueError("Test case is missing required product_description")
    
    result.append({"text": test_case["product_description"]})
    
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

def llm_with_tool_reasoning(
    bedrock_client,
    messages,
    tools,
    model_id,
    max_tokens=8000,
    temperature=0.2,
    system_prompt=SYSTEM_PROMPT,
):
    """Invoke Claude with tool use and reasoning using the converse API"""
    tool_config = {"tools": tools}
    reasoning_config = {}

    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=[{"text": system_prompt}],
            toolConfig=tool_config,
            additionalModelRequestFields=reasoning_config,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )

        assistant_message = response.get("output", {}).get("message", {})

        if "content" in assistant_message:
            filtered_content = []
            for content_block in assistant_message["content"]:
                if "SDK_UNKNOWN_MEMBER" not in str(content_block):
                    filtered_content.append(content_block)
            assistant_message["content"] = filtered_content

        has_tool_calls = False
        tool_use_blocks = []

        if response.get("stopReason") == "tool_use":
            has_tool_calls = True
            for content_block in assistant_message["content"]:
                if "toolUse" in content_block.keys():
                    tool_use_blocks.append(content_block["toolUse"])
        
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
    """Process tool calls using local embeddings"""
    tool_results = []

    for tool_block in tool_use_blocks:
        tool_name = tool_block.get("name")
        tool_input = tool_block.get("input", {})
        tool_id = tool_block.get("toolUseId")

        try:
            # Import local tools
            if import_local_tools():
                import tools
                if hasattr(tools, tool_name):
                    function = getattr(tools, tool_name)
                    tool_result = function(**tool_input)
                else:
                    tool_result = {"error": f"Unknown tool: {tool_name}"}
            else:
                # Fallback behavior for local embeddings
                tool_result = handle_tool_call_fallback(tool_name, tool_input)

            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_id,
                    "content": [{"json": tool_result}],
                }
            })
        except Exception as e:
            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_id,
                    "content": [{"json": {"error": f"Error processing tool: {str(e)}"}}],
                }
            })

    return tool_results

def handle_tool_call_fallback(tool_name, tool_input):
    """Fallback tool handling when local embeddings are not available"""
    if tool_name == "semantic_search":
        return {
            "total_hits": 1,
            "results": [{
                "eccn_code": "5A991.b.1",
                "description": "Telecommunications switching equipment (basic networking equipment without encryption)",
                "category": "5",
                "product_group": "Telecommunications"
            }],
            "note": "Using fallback classification - local embeddings not available"
        }
    else:
        return {"error": f"Tool {tool_name} not available in fallback mode"}

def extract_final_text_from_response(response):
    """Extract text content from Claude's final response"""
    text_content = []
    for block in response.get("content", []):
        if "text" in block:
            text_content.append(block.get("text", ""))
    return " ".join(text_content)

def lambda_handler(event, context):
    """AWS Lambda handler for ECCN classification with local embeddings"""
    try:
        # Check for multipart form data
        multipart_data = parse_multipart_form_data(event)
        
        if multipart_data:
            # Format 1: Multipart form data
            fields = multipart_data.get('fields', {})
            files = multipart_data.get('files', [])
            
            # Process uploaded files (PDF only)
            file_description, processed_files = process_uploaded_files(files)
            
            form_description = fields.get('product_description', '').strip()
            product_description = form_description if form_description else file_description
            
            # For PDF-only mode, allow empty product_description if we have PDF files
            if not product_description and not processed_files:
                raise ValueError("Either product_description or PDF files are required")
            
            # If no text description but we have PDFs, use a default description
            if not product_description and processed_files:
                product_description = "Product information extracted from uploaded PDF files"
            
            test_case = {
                "product_description": product_description,
                "pdf_files": processed_files
            }
            
            original_message = build_user_message_from_direct_input(test_case)
            
            max_tool_iterations = int(fields.get("max_tool_iterations", "5"))
            bedrock_model_id = fields.get("bedrock_model_id", 
                                        os.environ.get("DEFAULT_BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"))
            max_tokens = int(fields.get("max_tokens", "8000"))
            temperature = float(fields.get("temperature", "0.3"))
            
        elif "test_case" in event:
            # Format 2: Direct JSON input
            test_case = event.get("test_case", {})
            if not test_case.get("product_description"):
                raise ValueError("test_case.product_description is required")
            
            original_message = build_user_message_from_direct_input(test_case)
            
            max_tool_iterations = event.get("max_tool_iterations", 5)
            bedrock_model_id = event.get(
                "bedrock_model_id", 
                os.environ.get("DEFAULT_BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
            )
            max_tokens = event.get("max_tokens", 8000)
            temperature = event.get("temperature", 0.3)
            
        else:
            # Basic GET request for health check
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "ECCN Classification API with Local Embeddings",
                    "version": "2.0.0",
                    "features": [
                        "Local ECCN embeddings (no OpenSearch)",
                        "PDF-only input support",
                        "Round 3 optimized system prompt",
                        "Claude 3.7 Sonnet integration"
                    ],
                    "status": "ready"
                })
            }

        # Initialize Bedrock client
        bedrock_client = boto3.client("bedrock-runtime")

        # Define tools
        if import_local_tools():
            from tools import get_tools_definition
            tools = get_tools_definition()
        else:
            tools = get_tools_definition_fallback()

        # Initialize conversation
        messages = [{"role": "user", "content": original_message}]

        all_responses = []
        has_tool_calls = False
        tool_use_blocks = []
        stop_reason = None

        # Main loop for tool reasoning
        for iteration in range(max_tool_iterations + 1):
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

            messages.append(assistant_response)
            all_responses.append(assistant_response)

            if not has_tool_calls or iteration == max_tool_iterations:
                break

            tool_results = process_tool_calls(tool_use_blocks)
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
                        all_responses[-1] if all_responses else assistant_response
                    ),
                    "has_additional_tool_calls": has_tool_calls,
                    "embeddings_type": "Local (no OpenSearch)",
                    "prompt_version": "Round 3 Optimized"
                },
                default=str,
            ),
        }

        return final_response

    except ValueError as ve:
        return {"statusCode": 400, "body": json.dumps({"error": str(ve)})}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}