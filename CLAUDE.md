# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an ECCN (Export Control Classification Number) intelligent analysis system built on AWS serverless architecture. The system processes PDF technical documents and classifies products according to US export control regulations using AI-powered analysis. The current production system achieves **90.7% accuracy** and **100% system stability** across **54 comprehensive test cases** (100% ground truth coverage).

## Core Architecture

### Two-Lambda Pipeline (Current Production)
```
PDF Upload → PDF Parser Lambda → S3 Storage → Main Classifier Lambda → ECCN Results
           (PyMuPDF4LLM)      (JSON content)   (Claude 3.7 + Tool Enhancement)
```

**Current System Performance (Latest Test Results)**:
- **Stack Name**: eccn-two-lambda-pipeline
- **Overall Accuracy**: 90.7% (49/54 correct classifications)
- **System Stability**: 100% (all tests complete successfully)
- **Processing Time**: ~16.59s average per classification
- **Total Processing Time**: 895.6 seconds for 54 tests
- **Ground Truth Coverage**: 100% (54/54 products from Product_proposal_55.xlsx)

### Classification Strategy
1. **Mouser API Direct**: Direct product lookup (37/39 correct, 94.9% accuracy)
2. **Complete Pipeline**: PDF analysis + AI classification (12/15 correct, 80.0% accuracy)

**Pipeline Components**:
- **PDF Parser Lambda**: PyMuPDF4LLM content extraction → S3 storage
- **Main Classifier Lambda**: Bedrock Claude 3.7 + embeddings + external tools
- **Tool Enhancement**: Mouser API integration + WebSearch verification
- **RAG System**: Real ECCN embeddings with cosine similarity search
- **Prompt Engineering**: Specialized classification logic with contextual rules

## Common Development Commands

### Infrastructure Deployment (Pulumi)
```bash
cd pulumi/

# Deploy the complete two-lambda system
pulumi up --yes

# Check deployment status and get Lambda URLs
pulumi stack output pdfParserUrl
pulumi stack output mainClassifierUrl

# Destroy infrastructure
pulumi down --yes
```

### Testing and Validation
```bash
cd pulumi/

# Run comprehensive test of all 54 PDFs (recommended)
python complete_hardcoded_test.py

# Monitor test progress
python monitor_test_progress.py

# Quick single test
curl -X POST $(pulumi stack output pdfParserUrl) \
  -H "Content-Type: multipart/form-data" \
  -F "file=@../src/sagemaker/data/EKI-5525I-AE/EKI-5525I20150714145223.pdf" \
  -F "product_model=EKI-5525I-AE"
```

### Local Development
```bash
# Install core dependencies
pip install boto3 pymupdf4llm pulumi pulumi-aws requests numpy scikit-learn

# Local testing with sample data
cd src/sagemaker/
python handler.py

# Test specific components
cd src/lambdas/find_eccn/
python handler_local_embeddings.py
```

### AWS Lambda Operations
```bash
# View Lambda logs
aws logs tail /aws/lambda/eccn-two-lambda-pipeline-pdfparser --follow
aws logs tail /aws/lambda/eccn-two-lambda-pipeline-main --follow

# Check function configuration
aws lambda get-function --function-name eccn-two-lambda-pipeline-main
```

## Key Technology Stack

### Core Technologies
- **AWS Lambda**: Two-function serverless architecture
  - PDF Parser: Python 3.12 + PyMuPDF4LLM layer (1024MB, 900s timeout)
  - Main Classifier: Python 3.10 + embeddings layer (2048MB, 900s timeout)
- **AWS Bedrock**: Claude 3.7 Sonnet model (`us.anthropic.claude-3-7-sonnet-20250219-v1:0`)
- **PyMuPDF4LLM**: PDF content parsing and markdown conversion
- **Pulumi**: Infrastructure as Code deployment
- **S3**: Document storage and pipeline coordination

### AI/ML Components
- **RAG System**: Real ECCN embeddings (`data.pkl`) with cosine similarity
- **Tool Enhancement**: Mouser API (API key: 773b916e-0f6c-4a86-a896-b3b435be5389)
- **WebSearch Integration**: Cross-validation with external sources
- **Content-Based Analysis**: Technical specification extraction without naming patterns
- **Specialized Prompts**: Context-aware classification rules in `prompts.py`

## Testing Framework

### Ground Truth Dataset
- **Test Data**: `src/sagemaker/data/` - 54 real industrial networking equipment PDFs
- **Reference Data**: `src/sagemaker/Product_proposal_55.xlsx` - 54 product records
- **Test Coverage**: 54/54 products (100% coverage)

### Current Performance Metrics
- **Overall Accuracy**: 90.7% (49/54 correct)
- **Mouser Direct**: 94.9% accuracy (37/39 correct)
- **Complete Pipeline**: 80.0% accuracy (12/15 correct)
- **System Stability**: 100% (zero timeouts or failures)

### ECCN Classification Distribution (54 Test Cases)
- **5A991** (Industrial): 24 products (44.4%)
- **EAR99** (Commercial): 13 products (24.1%)
- **4A994** (Management): 8 products (14.8%)
- **5A991.b.1** (High-speed): 4 products (7.4%)
- **5A991.b** (Enhanced Industrial): 3 products (5.6%)
- **5A992.c** (High-end): 2 products (3.7%)
- **5A002** (Security): 1 product (1.9%)

## API Usage

### Two-Lambda Pipeline API
```bash
# Get Lambda URLs
PDF_PARSER_URL=$(pulumi stack output pdfParserUrl)
MAIN_CLASSIFIER_URL=$(pulumi stack output mainClassifierUrl)

# Step 1: PDF Processing
curl -X POST $PDF_PARSER_URL \
  -H "Content-Type: multipart/form-data" \
  -F "file=@product_datasheet.pdf" \
  -F "product_model=EKI-5525I"

# Step 2: Classification (using returned S3 key)
curl -X POST $MAIN_CLASSIFIER_URL \
  -H "Content-Type: application/json" \
  -d '{
    "s3_key": "parsed/pdf_20250729_123456_abcd.json",
    "product_model": "EKI-5525I",
    "debug": true
  }'
```

### Response Format
```json
{
  "statusCode": 200,
  "body": {
    "success": true,
    "classification": {
      "eccn_code": "5A991",
      "confidence": "high",
      "method": "mouser_api_direct",
      "reasoning": "Direct match from Mouser API database",
      "processing_time": "7.8s"
    }
  }
}
```

## Critical Development Requirements

### Core Constraints (MANDATORY)
1. **Content-Based Classification Only**: Must analyze actual PDF technical content, not product naming patterns
2. **PDF-Only Testing**: All testing must use actual PDF files from `src/sagemaker/data/`
3. **Lambda Deployment**: All handlers must be deployed to AWS Lambda for validation
4. **Two-Lambda Architecture**: Maintain separate PDF processing and classification functions

### Classification Methodology
- **Technical Specification Analysis**: Temperature ranges, power requirements, data rates
- **Industrial Grade Detection**: DC power supply, extended temperature, DIN-rail mounting
- **High-Speed Detection**: Gigabit capabilities, switching capacity, throughput
- **Management Function Detection**: SNMP, web interfaces, network management
- **Security Feature Detection**: Encryption, VPN, authentication protocols

## Key Files for Development

### Core System Files
- **`pulumi/Pulumi.yaml`**: Infrastructure configuration
- **`pulumi/lambda_function.py`**: Main classifier handler
- **`pulumi/pdf_parser.py`**: PDF parser handler
- **`pulumi/prompts.py`**: Classification logic and rules
- **`pulumi/tools.py`**: Mouser API and WebSearch integration
- **`pulumi/websearch.py`**: Web search functionality

### Testing and Data
- **`pulumi/complete_hardcoded_test.py`**: Comprehensive test suite (54 cases)
- **`src/sagemaker/Product_proposal_55.xlsx`**: Ground truth dataset
- **`src/sagemaker/data/`**: Real PDF test files (54 products)
- **`pulumi/data.pkl`**: ECCN embeddings data

### Development History
- **v8.0**: Pattern-based method (36.2% accuracy, 45 test cases)
- **v9.0**: Initial specification-based approach (20% accuracy)
- **v3.2 Pipeline**: Current production system (90.7% accuracy, 100% stability, 54 test cases)

### Recent Improvements (Latest Testing Cycle)
- **Expanded Coverage**: Successfully incorporated 9 additional PDF test cases
- **Complete Ground Truth**: Achieved 100% coverage of all products in ground truth dataset
- **Maintained Stability**: System continues to demonstrate 100% reliability across all 54 cases
- **Performance Insights**: 5 remaining classification errors identified for future optimization
  - 3 Mouser API errors (EKI-2741F-BE, EKI-2541M-BE, EKI-5728)
  - 2 Complete Pipeline errors (EKI-2728M-BE, EKI-2541SI-BE)

## Troubleshooting

### Common Issues
- **Lambda Timeout**: Resolved in current version (100% stability)
- **API Rate Limits**: Mouser API key configured, 1s delays between requests
- **Classification Errors**: Check `complete_hardcoded_test_results_*.json` for error analysis

### Debugging Tools
```bash
# Real-time test monitoring
python monitor_test_progress.py

# Check latest test results
ls -la complete_hardcoded_test_results_*.json

# AWS CloudWatch logs
aws logs tail /aws/lambda/eccn-two-lambda-pipeline-main --follow
```

## Security and Cost

### Security Configuration
- **IAM Role**: `AIWebSearch-AgentLambdaRole-7AAqt7ZLhLi0` (principle of least privilege)
- **HTTPS**: All endpoints enforce HTTPS with CORS
- **S3 Encryption**: Server-side encryption (AES256)

### Cost Analysis (per classification)
- **Lambda Execution**: ~$0.01-0.03
- **Bedrock Claude**: ~$0.02-0.08  
- **S3 Storage**: ~$0.001 per GB/month
- **Total Cost**: ~$0.04-0.12 per analysis