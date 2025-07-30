import json
import base64
import pymupdf4llm
import fitz
import os
import uuid
import re
from datetime import datetime

# 只在 Lambda 環境中導入 boto3
try:
    import boto3
    # 使用環境變數的 AWS 憑證
    s3_client = boto3.client(
        's3',
        region_name=os.environ.get('CUSTOM_AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False

# 嘗試導入 requests (需要在 layer 中)
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'eccn-two-lambda-pipeline-data-us-east-1')

# Main Classifier Lambda URL
MAIN_CLASSIFIER_URL = os.environ.get(
    'MAIN_CLASSIFIER_URL', 
    'https://xsdknytlo2bjecf2zbypoeybcm0razel.lambda-url.us-east-1.on.aws/'
)

def extract_product_model_from_text(text):
    """從PDF文本中提取產品型號"""
    # 優先查找文件名模式 (如: EKI-5729FI-MB)
    model_patterns = [
        r'\b(EKI-\w+(?:-\w+)*)\b',  # EKI-xxxx 系列
        r'\b(TN-\w+(?:-\w+)*)\b',   # TN-xxxx 系列  
        r'\b([A-Z]{2,4}-\d+\w*(?:-\w+)*)\b',  # 通用型號模式
        r'(?:Product|Model|Part)[\s:]*([A-Z0-9-]+)',  # Product: xxxx
    ]
    
    for pattern in model_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0]
    
    return None

def call_main_classifier(s3_key, product_model):
    """呼叫 Main Classifier Lambda 進行 ECCN 分類"""
    if not REQUESTS_AVAILABLE:
        return {
            "success": False,
            "error": "requests module not available"
        }
    
    try:
        payload = {
            "s3_key": s3_key,
            "product_model": product_model,
            "debug": True
        }
        
        response = requests.post(
            MAIN_CLASSIFIER_URL,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=120  # 120秒超時（足夠處理複雜分類）
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"Main classifier returned {response.status_code}",
                "details": response.text[:500]
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to call main classifier: {str(e)}"
        }

def lambda_handler(event, context):
    # 解析 API Gateway proxy event
    try:
        if event.get('isBase64Encoded'):
            body = base64.b64decode(event['body'])
        else:
            body = event['body'].encode('utf-8')
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Failed to decode body: {str(e)}'})
        }

    # 嘗試解析 multipart/form-data
    content_type = event['headers'].get('content-type') or event['headers'].get('Content-Type')
    if not content_type or 'multipart/form-data' not in content_type:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Content-Type must be multipart/form-data'})
        }
    boundary = content_type.split('boundary=')[-1]
    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]
    boundary = boundary.encode('utf-8')
    
    # 簡單分割取得 PDF 檔案內容和product_model參數
    parts = body.split(b'--' + boundary)
    pdf_bytes = None
    product_model_param = None
    
    for part in parts:
        if b'Content-Disposition' in part:
            if b'filename=' in part:
                # PDF file part
                pdf_start = part.find(b'\r\n\r\n')
                if pdf_start != -1:
                    pdf_bytes = part[pdf_start+4:]
            elif b'name="product_model"' in part:
                # product_model parameter
                param_start = part.find(b'\r\n\r\n')
                if param_start != -1:
                    product_model_param = part[param_start+4:].decode('utf-8').strip()
    
    if not pdf_bytes:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No PDF file found in form-data'})
        }
    
    # 用 pymupdf4llm 解析 PDF
    try:
        if pdf_bytes is None:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'PDF bytes is None - failed to extract PDF from multipart data'})
            }
        
        # 創建臨時檔案給 pymupdf4llm 使用
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(pdf_bytes)
            tmp_file.flush()
            # 使用檔案路徑而非 bytes stream
            text = pymupdf4llm.to_markdown(tmp_file.name)
        
        # 清理臨時檔案
        try:
            os.unlink(tmp_file.name)
        except:
            pass
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Failed to extract text: {str(e)}'})
        }
    
    # 使用用戶提供的product_model參數，如果沒有提供則從PDF提取
    if product_model_param:
        product_model = product_model_param
    else:
        product_model = extract_product_model_from_text(text)
    
    # 嘗試上傳到 S3
    if S3_AVAILABLE:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_id = str(uuid.uuid4())[:8]
            s3_key = f"parsed/pdf_{timestamp}_{file_id}.json"
            
            parsed_data = {
                "content": text,
                "timestamp": timestamp,
                "parser_used": "pymupdf4llm",
                "product_model": product_model,
                "user_provided_model": product_model_param,
                "extracted_model": extract_product_model_from_text(text)
            }
            
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=s3_key,
                Body=json.dumps(parsed_data, ensure_ascii=False),
                ContentType='application/json'
            )
            
            #  關鍵步驟：自動呼叫 Main Classifier，傳入完整的PDF內容和產品型號
            if product_model and REQUESTS_AVAILABLE:
                classification_result = call_main_classifier(s3_key, product_model)
                
                if classification_result.get('success'):
                    # 成功：返回完整的分類結果（單次 curl 完成）
                    return {
                        'statusCode': 200,
                        'body': json.dumps({
                            'success': True,
                            'pipeline_mode': 'single_curl_complete',
                            'pdf_processing': {
                                's3_bucket': BUCKET_NAME,
                                's3_key': s3_key,
                                'extracted_text_length': len(text),
                                'product_model_detected': product_model
                            },
                            'eccn_classification': classification_result.get('classification', {}),
                            'data_source_details': classification_result.get('classification', {}).get('data_source_details', {}),
                            'message': f' Complete pipeline: PDF → ECCN classification for {product_model}'
                        }),
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
                else:
                    # 分類失敗：返回解析結果 + 錯誤信息
                    return {
                        'statusCode': 200,
                        'body': json.dumps({
                            'success': True,
                            'pipeline_mode': 'partial_complete',
                            'pdf_processing': {
                                's3_bucket': BUCKET_NAME,
                                's3_key': s3_key,
                                'extracted_text_length': len(text),
                                'product_model_detected': product_model
                            },
                            'eccn_classification_error': classification_result.get('error'),
                            'message': f' PDF parsed successfully but ECCN classification failed for {product_model}'
                        }),
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        }
                    }
            else:
                # 無法呼叫 Main Classifier：只返回解析結果
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'success': True,
                        'pipeline_mode': 'parse_only',
                        'pdf_processing': {
                            's3_bucket': BUCKET_NAME,
                            's3_key': s3_key,
                            'extracted_text_length': len(text),
                            'product_model_detected': product_model
                        },
                        'message': f'PDF parsed. Product model: {product_model or "Not detected"}. Manual classification needed.',
                        'manual_classification_url': f'{MAIN_CLASSIFIER_URL}',
                        'manual_payload': {
                            's3_key': s3_key,
                            'product_model': product_model or 'Unknown',
                            'debug': True
                        }
                    }),
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    }
                }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': f'S3 upload failed: {str(e)}'})
            }
    
    # 備援：直接返回解析結果
    return {
        'statusCode': 200,
        'body': json.dumps({
            'extracted_text': text,
            'text_length': len(text),
            'product_model_detected': product_model,
            'message': 'PDF parsed successfully (direct mode, no S3)'
        }),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }