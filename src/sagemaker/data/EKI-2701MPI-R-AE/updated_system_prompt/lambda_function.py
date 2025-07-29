#!/usr/bin/env python3
"""
ECCN Classification Handler - System Prompt Based Analysis
Direct AI analysis of PDF content without feature extraction
"""

import json
import boto3
import logging
import os
from typing import Dict, Any, Optional

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SystemPromptECCNClassifier:
    """System prompt based ECCN classifier using AI to analyze PDF content directly"""
    
    def __init__(self):
        # Use environment variables for credentials
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )
        
    def get_eccn_classification_prompt(self, pdf_content: str) -> str:
        """Generate system prompt for ECCN classification based on ground truth analysis"""
        
        system_prompt = f"""
You are an expert in US Export Control Classification Numbers (ECCN) for industrial networking equipment. 

Analyze the provided PDF technical specification and classify it into the correct ECCN category based on these rules:

## ECCN Classification Rules (Based on Ground Truth Analysis):

### EAR99 - Commercial/Consumer Grade (35.2% of products)
**Characteristics:**
- Commercial grade equipment for office/business use
- Operating temperature: 0°C to 60°C (32°F to 140°F)  
- Basic unmanaged switches
- No advanced industrial features
- Standard commercial environment rating

### 5A991 - Standard Industrial Grade (29.6% of products)
**Characteristics:**
- Industrial Ethernet switches for factory automation
- Operating temperature: -40°C to 75°C (-40°F to 167°F)
- DIN rail mounting capability
- Industrial environment protection (IP30/IP40)
- Basic industrial networking features
- NOT high-performance or enhanced security

### 5A991.b - Enhanced Industrial Grade (9.3% of products)
**Characteristics:**
- Enhanced industrial switches with security/redundancy features
- Operating temperature: -40°C to 75°C
- Advanced security features (VLANs, access control, authentication)
- Redundancy features (power, network paths)
- Enhanced management capabilities

### 5A991.b.1 - High-Performance Industrial (7.4% of products)  
**Characteristics:**
- **High-performance industrial equipment with specialized capabilities**
- **PoE Extenders with high power delivery (50-60W)**
- **HDBaseT or similar extension technologies**
- **Gigabit speeds (1000 Mbps) or higher**
- **8+ Gigabit Ethernet ports on switches**
- **Media converters with long-distance capability (>1km)**
- **Special industrial designations indicating enhanced features**
- Industrial temperature range: -40°C to 75°C
- **Key indicators**: "Extender", "HDBaseT", "60W PoE", "Media Converter", "8GE", "LI" (enhanced industrial)
- **Examples**: PoE extenders, media converters, high-port-density gigabit switches, long-reach industrial equipment

### 4A994 - Management Equipment (13.0% of products)
**Characteristics:**
- Managed switches with centralized management capabilities
- SNMP management protocols
- Network management software
- Centralized configuration and monitoring
- May have "M" (Management) or "MI" (Management Industrial) in model name

## Analysis Instructions:
1. Read the entire PDF content carefully
2. Focus on technical specifications, not product naming patterns
3. Identify key indicators: temperature range, management features, performance specs, security features
4. **Special consideration for 5A991.b.1**: Look for high-performance indicators:
   - **PoE Extender functionality with high power (50-60W)**
   - **HDBaseT or similar extension technologies** 
   - **Gigabit (1000 Mbps) speeds or higher**
   - **8+ Gigabit Ethernet ports**
   - **Media conversion capabilities with long-distance support**
   - **"LI" designation in product name** (indicates Long Industrial/enhanced features)
   - **"Extender" in product description**
   - Power consumption significantly above basic switches (>20W)
   - Advanced industrial features beyond basic switching
5. Apply the classification rules above in priority order
6. Provide your reasoning based on specific technical details found

## Response Format:
{{
  "eccn_code": "selected_eccn",
  "confidence": "High/Medium/Low", 
  "reasoning": "Detailed explanation citing specific technical specifications from the PDF",
  "key_indicators": ["list", "of", "key", "technical", "features", "found"]
}}

## PDF Content to Analyze:
{pdf_content}

Classify this product based on its technical specifications only. Do NOT use product naming patterns for classification.
"""
        return system_prompt
    
    def classify_with_bedrock(self, pdf_content: str) -> Dict[str, Any]:
        """Use Bedrock Claude to classify ECCN based on PDF content"""
        
        try:
            # Prepare the prompt
            prompt = self.get_eccn_classification_prompt(pdf_content)
            
            # Prepare Bedrock request
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0.1,  # Low temperature for consistent classification
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            }
            
            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20241022-v2:0",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read().decode('utf-8'))
            claude_response = response_body['content'][0]['text']
            
            # Try to parse JSON response
            try:
                # Extract JSON from Claude's response - find the first complete JSON object
                import re
                # Look for JSON object with proper braces matching
                start = claude_response.find('{')
                if start != -1:
                    # Find matching closing brace
                    brace_count = 0
                    end = start
                    for i, char in enumerate(claude_response[start:], start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end = i + 1
                                break
                    
                    json_str = claude_response[start:end]
                    classification_result = json.loads(json_str)
                else:
                    # Fallback parsing
                    classification_result = {
                        "eccn_code": "EAR99",
                        "confidence": "Low",
                        "reasoning": "Could not find JSON in AI response",
                        "key_indicators": []
                    }
            except:
                # Fallback if JSON parsing fails
                classification_result = {
                    "eccn_code": "EAR99", 
                    "confidence": "Low",
                    "reasoning": f"AI response parsing failed. Raw response: {claude_response[:200]}...",
                    "key_indicators": []
                }
            
            # Add method info
            classification_result['method'] = 'system_prompt_ai_analysis'
            classification_result['ai_response'] = claude_response
            
            return classification_result
            
        except Exception as e:
            logger.error(f"Bedrock classification error: {str(e)}")
            return {
                "eccn_code": "EAR99",
                "confidence": "Low", 
                "reasoning": f"Classification failed due to error: {str(e)}",
                "method": "system_prompt_ai_analysis",
                "error": str(e),
                "key_indicators": []
            }

# Lambda handler function
def lambda_handler(event, context):
    """Main Lambda handler for system prompt based ECCN classification"""
    
    try:
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)
        
        s3_key = body.get('s3_key')
        debug = body.get('debug', False)
        
        if not s3_key:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing s3_key parameter',
                    'success': False
                })
            }
        
        # Initialize classifier
        classifier = SystemPromptECCNClassifier()
        
        # Get PDF content from S3
        bucket_name = os.environ.get('S3_BUCKET_NAME', 'eccn-enhanced-pipeline-data-us-east-1')
        
        try:
            response = classifier.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            pdf_content = response['Body'].read().decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading from S3: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': f'Failed to read PDF content from S3: {str(e)}',
                    'success': False
                })
            }
        
        # Classify using AI system prompt approach
        classification_result = classifier.classify_with_bedrock(pdf_content)
        
        # Prepare response
        response_data = {
            'classification': classification_result,
            'success': True
        }
        
        if debug:
            response_data['debug_info'] = {
                'version': 'system_prompt_v1.0',
                'method': 'ai_direct_analysis',
                'pdf_length': len(pdf_content),
                'approach': 'system_prompt_based_classification',
                'no_feature_extraction': True,
                'ai_model': 'claude-3-7-sonnet-20241022-v2'
            }
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': f'Classification failed: {str(e)}',
                'success': False
            })
        }

if __name__ == "__main__":
    # Test with sample data
    test_event = {
        'body': {
            's3_key': 'test_document.txt',
            'debug': True
        }
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))