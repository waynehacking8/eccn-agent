#!/usr/bin/env python3
"""
完整ECCN分類Pipeline - 單curl完成所有流程
設計流程：
1. 優先Mouser API直接查詢 → 找到則直接返回
2. 查詢不到 → 同時執行：PDF特徵→Mouser相似產品查詢 + WebSearch交叉驗證
3. 將所有結果給LLM綜合決策
4. 顯示完整資料來源
"""

import json
import logging
import boto3
import time
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

# AWS服務設定 - 從環境變數讀取
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('CUSTOM_AWS_REGION', 'us-east-1')

# S3和Bedrock配置
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME', 'eccn-two-lambda-pipeline-data-us-east-1')
BEDROCK_MODEL_ID = os.environ.get('DEFAULT_BEDROCK_MODEL_ID', "us.anthropic.claude-3-7-sonnet-20250219-v1:0")

class CompletePipelineECCNClassifier:
    """完整Pipeline ECCN分類器"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # AWS服務初始化
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
    
    def _normalize_eccn_format(self, eccn_code: str) -> str:
        """統一ECCN代碼格式 - 最後一段字母後綴轉小寫"""
        if not eccn_code or not isinstance(eccn_code, str):
            return "EAR99"
        
        eccn_code = eccn_code.strip()
        
        # EAR99 保持大寫
        if eccn_code.upper() == 'EAR99':
            return 'EAR99'
        
        # 處理其他ECCN代碼，只將最後的字母後綴轉小寫
        import re
        # 匹配格式如：5A992.C → 5A992.c, 5A991.B.1 → 5A991.b.1
        eccn_code = re.sub(r'\.([A-Z])(\.[0-9]+)?$', lambda m: f'.{m.group(1).lower()}{m.group(2) or ""}', eccn_code)
        
        return eccn_code
    
    def _setup_logging(self):
        """設定日誌記錄"""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除現有handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # 設定新handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def classify_eccn(self, s3_key: str, product_model: str, debug: bool = False) -> Dict[str, Any]:
        """執行完整Pipeline ECCN分類"""
        start_time = time.time()
        
        self.logger.info(f"🚀 開始完整Pipeline ECCN分類: {product_model}")
        
        try:
            # 步驟1: 優先Mouser API直接查詢
            self.logger.info("🔍 步驟1: Mouser API直接查詢...")
            mouser_direct_result = self._mouser_direct_search(product_model)
            
            if mouser_direct_result and mouser_direct_result.get('eccn_code'):
                # 找到直接匹配，立即返回
                self.logger.info(f"✅ Mouser API直接找到: {mouser_direct_result.get('eccn_code')}")
                return self._create_success_response({
                    'eccn_code': mouser_direct_result.get('eccn_code'),
                    'confidence': mouser_direct_result.get('confidence', 'high'),
                    'method': 'mouser_api_direct',
                    'reasoning': f'Mouser API直接查詢結果: {mouser_direct_result.get("reasoning", "")}',
                    'data_sources': {
                        'primary_source': 'mouser_api_direct',
                        'mouser_direct': '✅ 成功',
                        'pdf_feature_analysis': '❌ 未執行（已找到直接匹配）',
                        'mouser_similar_search': '❌ 未執行（已找到直接匹配）',
                        'websearch_validation': '❌ 未執行（已找到直接匹配）',
                        'llm_decision': '❌ 未執行（已找到直接匹配）'
                    },
                    'processing_time': f"{time.time() - start_time:.2f}s"
                })
            
            # 步驟2: Mouser API未找到，直接進入技術規格分析
            self.logger.info("❌ Mouser API直接查詢未找到，開始PDF技術規格分析...")
            
            # 2.1 獲取PDF技術內容
            pdf_content = self._get_pdf_content(s3_key)
            if not pdf_content:
                return self._create_error_response("無法獲取PDF內容", s3_key)
            
            # 2.2 提取PDF技術規格
            self.logger.info("📋 提取PDF技術規格...")
            technical_specs = self._extract_technical_specifications(pdf_content)
            
            # 2.3 基於技術規格執行Mouser相似產品查詢
            self.logger.info("🔍 基於技術規格執行Mouser相似產品查詢...")
            mouser_similar_result = self._mouser_similar_search(pdf_content, product_model)
            websearch_result = self._websearch_validation(product_model)
            
            # 步驟3: 基於技術規格的LLM分類決策 
            self.logger.info("🤖 基於技術規格進行LLM分類決策...")
            final_classification = self._specification_based_classification(
                pdf_content, 
                product_model,
                technical_specs,
                mouser_similar_result,
                websearch_result
            )
            
            # 格式化最終結果
            final_result = self._format_comprehensive_response(
                final_classification,
                mouser_similar_result,
                websearch_result,
                start_time,
                debug
            )
            
            self.logger.info(f"✅ Pipeline完成: {final_result.get('eccn_code')} ({time.time() - start_time:.2f}s)")
            return self._create_success_response(final_result)
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline失敗: {str(e)}")
            # 最終失敗保護 - 永遠不返回null
            failsafe_result = self._get_failsafe_classification(product_model)
            return self._create_success_response(failsafe_result)
    
    def _extract_technical_specifications(self, pdf_content: str) -> Dict:
        """從PDF內容提取技術規格"""
        try:
            # 提取關鍵技術規格
            specs = {
                'temperature_range': self._extract_temperature_range(pdf_content),
                'power_specifications': self._extract_power_specs(pdf_content),
                'management_features': self._extract_management_features(pdf_content),
                'port_information': self._extract_port_info(pdf_content),
                'performance_specs': self._extract_performance_specs(pdf_content),
                'security_features': self._extract_security_features(pdf_content),
                'environmental_ratings': self._extract_environmental_ratings(pdf_content)
            }
            return specs
        except Exception as e:
            self.logger.warning(f"⚠️ 技術規格提取失敗: {str(e)}")
            return {}
    
    def _extract_temperature_range(self, pdf_content: str) -> str:
        """提取工作溫度範圍"""
        import re
        # 查找溫度範圍模式
        temp_patterns = [
            r'(-?\d+)°C\s*(?:to|~|-)\s*\+?(\d+)°C',
            r'Operating Temperature[:\s]+(-?\d+)°C\s*(?:to|~|-)\s*\+?(\d+)°C',
            r'Temperature[:\s]+(-?\d+)°C\s*(?:to|~|-)\s*\+?(\d+)°C'
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, pdf_content, re.IGNORECASE)
            if match:
                return f"{match.group(1)}°C to +{match.group(2)}°C"
        return "Not specified"
    
    def _extract_power_specs(self, pdf_content: str) -> str:
        """提取電源規格"""
        import re
        # 查找電源規格
        if re.search(r'12V|24V|48V.*DC', pdf_content, re.IGNORECASE):
            return "DC Power"
        elif re.search(r'100-240V.*AC', pdf_content, re.IGNORECASE):
            return "AC Power"
        elif re.search(r'AC/DC|DC/AC', pdf_content, re.IGNORECASE):
            return "AC/DC Hybrid"
        return "Not specified"
    
    def _extract_management_features(self, pdf_content: str) -> str:
        """提取管理功能"""
        import re
        features = []
        if re.search(r'SNMP|Simple Network Management', pdf_content, re.IGNORECASE):
            features.append("SNMP")
        if re.search(r'Web.*GUI|Web.*Interface|HTTP', pdf_content, re.IGNORECASE):
            features.append("Web GUI")
        if re.search(r'CLI|Command.*Line|Telnet|SSH', pdf_content, re.IGNORECASE):
            features.append("CLI")
        if re.search(r'Unmanaged|No.*Management', pdf_content, re.IGNORECASE):
            return "Unmanaged"
        return ", ".join(features) if features else "Basic/Unmanaged"
    
    def _extract_security_features(self, pdf_content: str) -> str:
        """提取安全功能"""
        import re
        features = []
        if re.search(r'VPN|Virtual Private Network', pdf_content, re.IGNORECASE):
            features.append("VPN")
        if re.search(r'Encryption|Encrypt', pdf_content, re.IGNORECASE):
            features.append("Encryption")
        if re.search(r'Firewall|Access Control List|ACL', pdf_content, re.IGNORECASE):
            features.append("Firewall/ACL")
        if re.search(r'802\.1X|Authentication|AAA', pdf_content, re.IGNORECASE):
            features.append("Authentication")
        return ", ".join(features) if features else "None specified"
    
    def _extract_performance_specs(self, pdf_content: str) -> str:
        """提取性能規格"""
        import re
        # 查找交換容量
        capacity_match = re.search(r'(\d+)\s*Gbps|(\d+)\s*Mbps', pdf_content, re.IGNORECASE)
        if capacity_match:
            if capacity_match.group(1):
                return f"{capacity_match.group(1)} Gbps"
            else:
                return f"{capacity_match.group(2)} Mbps" 
        return "Not specified"
    
    def _extract_environmental_ratings(self, pdf_content: str) -> str:
        """提取環境保護等級"""
        import re
        if re.search(r'IP\d+|IP\s*\d+', pdf_content, re.IGNORECASE):
            ip_match = re.search(r'IP\s*(\d+)', pdf_content, re.IGNORECASE)
            if ip_match:
                return f"IP{ip_match.group(1)}"
        if re.search(r'DIN.*rail|DIN-rail', pdf_content, re.IGNORECASE):
            return "DIN-rail mounting"
        return "Not specified"
    
    def _get_failsafe_classification(self, product_model: str) -> Dict:
        """Final failsafe classification - ensures never returns null"""
        self.logger.info("🛡️ Activating failsafe classification...")
        
        # Conservative classification based on product type
        if not product_model:
            eccn_code = 'EAR99'  # Default to commercial grade
            reasoning = 'No product model provided - defaulting to commercial grade classification'
        else:
            model = product_model.upper()
            
            # Conservative technical assessment
            if 'EKI' in model:
                # Industrial networking equipment default
                eccn_code = '5A991'
                reasoning = f'Failsafe classification: {product_model} appears to be industrial networking equipment - conservative 5A991 classification'
            else:
                # Unknown product type - conservative commercial
                eccn_code = 'EAR99'
                reasoning = f'Failsafe classification: {product_model} - conservative commercial grade classification due to uncertainty'
        
        return {
            'eccn_code': self._normalize_eccn_format(eccn_code),
            'confidence': 'low',
            'method': 'failsafe_classification',
            'reasoning': reasoning,
            'data_sources': {
                'primary_source': 'failsafe_pattern_matching',
                'mouser_direct': '❌ 失敗',
                'pattern_matching': '❌ 失敗',
                'pdf_feature_analysis': '❌ 失敗',
                'mouser_similar_search': '❌ 失敗', 
                'websearch_validation': '❌ 失敗',
                'llm_decision': '❌ 失敗',
                'failsafe_protection': '✅ 已啟動'
            },
            'processing_time': '0.01s',
            'warning': 'This is a failsafe classification - manual review recommended'
        }

    def _mouser_direct_search(self, product_model: str) -> Optional[Dict]:
        """Mouser API直接查詢"""
        try:
            from tools import ECCNToolEnhancer
            tool_enhancer = ECCNToolEnhancer(self.logger)
            
            result = tool_enhancer.search_mouser_eccn(product_model)
            
            if result and result.get('eccn_code'):
                self.logger.info(f"✅ Mouser直接查詢成功: {result.get('eccn_code')}")
                return result
            else:
                self.logger.info("❌ Mouser直接查詢未找到匹配")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Mouser直接查詢失敗: {str(e)}")
            return None
    
    def _mouser_similar_search(self, pdf_content: str, product_model: str) -> Dict:
        """PDF feature-based Mouser similarity search"""
        try:
            self.logger.info("🔍 Mouser similarity analysis starting...")
            
            from mouser_algorithm import MouserSimilarityAnalyzer
            
            # Initialize analyzer
            analyzer = MouserSimilarityAnalyzer(logger=self.logger)
            
            # Run similarity analysis
            result = analyzer.analyze_similar_products(pdf_content, product_model)
            
            if result.get('status') == 'success':
                self.logger.info(f"✅ Mouser analysis successful: {result.get('eccn_code')} (confidence: {result.get('confidence')})")
                
                # Pass results to LLM for final decision
                return {
                    'status': 'success',
                    'eccn_suggestions': [result],
                    'similar_products_count': result.get('similar_products_count', 0),
                    'method': 'mouser_similarity',
                    'confidence': result.get('confidence'),
                    'reasoning': result.get('reasoning'),
                    'technical_analysis': result.get('technical_analysis'),
                    'eccn_distribution': result.get('eccn_distribution')
                }
            else:
                self.logger.info(f"❌ Mouser analysis failed: {result.get('error', 'Unknown error')}")
                return {
                    'status': 'no_results',
                    'eccn_suggestions': [],
                    'similar_products_count': 0,
                    'method': 'mouser_similarity',
                    'error': result.get('error')
                }
                
        except Exception as e:
            self.logger.error(f"⚠️ Mouser similarity analysis exception: {str(e)}")
            
            # Fallback to original method if enhanced fails
            try:
                self.logger.info("🔄 Falling back to original Mouser method...")
                from tools import MouserAPIClient
                mouser_client = MouserAPIClient(logger=self.logger)
                
                result = mouser_client.search_by_features(pdf_content, product_model)
                
                if result and result.get('eccn_code'):
                    return {
                        'status': 'success',
                        'eccn_suggestions': [result],
                        'similar_products_count': result.get('similar_products_count', 0),
                        'method': 'fallback_mouser_similarity'
                    }
                    
            except Exception as fallback_error:
                self.logger.error(f"⚠️ Fallback Mouser method also failed: {str(fallback_error)}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'eccn_suggestions': [],
                'method': 'enhanced_mouser_similarity'
            }
    
    def _websearch_validation(self, product_model: str) -> Dict:
        """WebSearch交叉驗證"""
        try:
            self.logger.info("🌐 執行WebSearch交叉驗證...")
            
            from websearch import ECCNWebSearcher
            web_searcher = ECCNWebSearcher(self.logger)
            
            results = web_searcher.search_eccn_information(product_model, "Advantech")
            
            if results:
                self.logger.info(f"✅ WebSearch找到 {len(results)} 個權威來源")
                return {
                    'status': 'success',
                    'eccn_suggestions': results[:5],  # 取前5個最相關
                    'sources_count': len(results),
                    'method': 'websearch_cross_validation'
                }
            else:
                self.logger.info("❌ WebSearch未找到相關來源")
                return {
                    'status': 'no_results',
                    'eccn_suggestions': [],
                    'sources_count': 0,
                    'method': 'websearch_cross_validation'
                }
                
        except Exception as e:
            self.logger.warning(f"⚠️ WebSearch交叉驗證失敗: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'eccn_suggestions': [],
                'method': 'websearch_cross_validation'
            }
    
    def _specification_based_classification(self, pdf_content: str, product_model: str,
                                  technical_specs: Dict, mouser_similar: Dict, websearch: Dict) -> Dict:
        """LLM綜合決策所有來源結果"""
        try:
            self.logger.info("🤖 執行LLM綜合決策...")
            
            # Gigabit檢測現在由prompts.py中的智能邏輯處理，不再使用硬編碼覆蓋
            
            # 安全功能檢測現在由prompts.py中的智能邏輯處理，不再使用硬編碼覆蓋
            self.logger.info("📋 所有分類邏輯現在統一由prompts.py智能處理，開始LLM綜合分析...")
            
            # 準備綜合上下文
            context = self._prepare_comprehensive_context(mouser_similar, websearch)
            
            # 導入技術規格提示詞
            from prompts import SYSTEM_PROMPT
            
            system_prompt = SYSTEM_PROMPT

            user_prompt = f"""
Product Model: {product_model}

Technical Specifications Analysis:
- Operating Temperature: {technical_specs.get('temperature_range', 'Not specified')}
- Power Specifications: {technical_specs.get('power_specifications', 'Not specified')}
- Management Features: {technical_specs.get('management_features', 'Not specified')}
- Security Features: {technical_specs.get('security_features', 'Not specified')}
- Performance Specifications: {technical_specs.get('performance_specs', 'Not specified')}
- Environmental Protection: {technical_specs.get('environmental_ratings', 'Not specified')}

MOUSER Similar Products Analysis:
{context['mouser_context']}

WEBSEARCH Cross-Validation:
{context['websearch_context']}

PDF Technical Content:
{pdf_content[:4000]}

Please perform ECCN classification based on technical specifications, with primary focus on temperature range and power specifications as key decision criteria.

Analysis Requirements:
1. If multiple sources consistently suggest the same ECCN, give high weight to that classification
2. Explain how each source influences your decision
3. Provide clear decision logic based on technical specifications
4. Focus on measurable technical parameters rather than product naming patterns

Please respond in JSON format."""

            # 調用Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 4000,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}]
                })
            )
            
            response_body = json.loads(response['body'].read())
            llm_text = response_body['content'][0]['text']
            
            # 解析JSON回應
            try:
                result = json.loads(llm_text)
                result['method'] = 'llm_comprehensive_decision'
                result['raw_llm_response'] = llm_text
                return result
            except json.JSONDecodeError:
                # 處理非JSON回應
                self.logger.warning("⚠️ LLM回應非JSON格式，嘗試解析")
                return self._parse_non_json_llm_response(llm_text, product_model)
                
        except Exception as e:
            self.logger.error(f"❌ LLM綜合決策失敗: {str(e)}")
            # 使用失敗保護分類
            failsafe = self._get_failsafe_classification(product_model)
            return {
                'eccn_code': failsafe['eccn_code'],
                'confidence': 'low',
                'reasoning': f'LLM綜合決策失敗，使用失敗保護分類: {failsafe["reasoning"]}',
                'method': 'llm_comprehensive_decision_failed_with_failsafe'
            }
    
    def _prepare_comprehensive_context(self, mouser_similar: Dict, websearch: Dict) -> Dict:
        """準備綜合上下文"""
        
        # Mouser相似產品上下文
        mouser_context = "Mouser相似產品分析:\n"
        if mouser_similar.get('status') == 'success':
            suggestions = mouser_similar.get('eccn_suggestions', [])
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    eccn = suggestion.get('eccn_code', 'N/A')
                    confidence = suggestion.get('confidence', 'N/A')
                    reasoning = suggestion.get('reasoning', '')[:200]
                    mouser_context += f"{i}. ECCN建議: {eccn} (信心度: {confidence})\n"
                    mouser_context += f"   理由: {reasoning}...\n"
            else:
                mouser_context += "未找到相似產品\n"
        else:
            mouser_context += f"查詢失敗: {mouser_similar.get('error', 'Unknown')}\n"
        
        # WebSearch權威來源上下文
        websearch_context = "WebSearch權威來源分析:\n"
        if websearch.get('status') == 'success':
            suggestions = websearch.get('eccn_suggestions', [])
            if suggestions:
                for i, suggestion in enumerate(suggestions, 1):
                    eccn = suggestion.get('eccn_code', 'N/A')
                    domain = suggestion.get('domain', 'N/A')
                    confidence = suggestion.get('confidence', 'N/A')
                    snippet = suggestion.get('snippet', '')[:150]
                    websearch_context += f"{i}. 來源: {domain}\n"
                    websearch_context += f"   ECCN建議: {eccn} (信心度: {confidence})\n"
                    websearch_context += f"   摘要: {snippet}...\n"
            else:
                websearch_context += "未找到權威來源\n"
        else:
            websearch_context += f"查詢失敗: {websearch.get('error', 'Unknown')}\n"
        
        return {
            'mouser_context': mouser_context,
            'websearch_context': websearch_context
        }
    
    def _parse_non_json_llm_response(self, llm_text: str, product_model: str) -> Dict:
        """解析非JSON格式的LLM回應"""
        import re
        
        # 嘗試提取ECCN代碼
        eccn_pattern = r'\b(EAR99|[0-9][A-Z][0-9]{3}(?:\.[a-z](?:\.[0-9]+)?)?)\b'
        eccn_matches = re.findall(eccn_pattern, llm_text, re.IGNORECASE)
        
        if eccn_matches:
            eccn_code = self._normalize_eccn_format(eccn_matches[0])
        else:
            # 使用失敗保護分類，不返回Unknown
            failsafe = self._get_failsafe_classification(product_model)
            eccn_code = self._normalize_eccn_format(failsafe['eccn_code'])
        
        return {
            'eccn_code': eccn_code,
            'confidence': 'medium',
            'reasoning': f'從LLM回應中解析: {llm_text[:500]}...',
            'method': 'llm_comprehensive_decision_parsed',
            'raw_llm_response': llm_text
        }
    
    def _format_comprehensive_response(self, classification: Dict, mouser_similar: Dict,
                                     websearch: Dict, start_time: float, debug: bool) -> Dict:
        """格式化綜合回應"""
        processing_time = time.time() - start_time
        
        # Determine the method based on which enhanced algorithm was used
        method = 'complete_pipeline'
        if mouser_similar.get('method') == 'enhanced_mouser_similarity':
            method = 'enhanced_mouser_similarity'
        elif mouser_similar.get('method') == 'fallback_mouser_similarity':
            method = 'fallback_mouser_similarity'
        
        result = {
            'eccn_code': classification.get('eccn_code', 'Unknown'),
            'confidence': classification.get('confidence', 'low'),
            'reasoning': classification.get('reasoning', ''),
            'method': method,
            'data_sources': {
                'primary_source': 'llm_comprehensive_decision',
                'mouser_direct': '❌ 未找到直接匹配',
                'pdf_feature_analysis': '✅ 已執行',
                'mouser_similar_search': '✅ 已執行' if mouser_similar.get('status') == 'success' else '❌ 失敗',
                'websearch_validation': '✅ 已執行' if websearch.get('status') == 'success' else '❌ 失敗',
                'llm_decision': '✅ 綜合決策完成'
            },
            'source_details': {
                'mouser_similar_products': mouser_similar.get('similar_products_count', 0),
                'websearch_sources': websearch.get('sources_count', 0),
                'mouser_influence': classification.get('mouser_influence', ''),
                'websearch_influence': classification.get('websearch_influence', ''),
                'mouser_method': mouser_similar.get('method', 'unknown')
            },
            'processing_time': f"{processing_time:.2f}s",
            'timestamp': datetime.now().isoformat(),
            'version': 'enhanced_pipeline_v1.1'
        }
        
        # Add enhanced algorithm details if available
        if mouser_similar.get('method') == 'enhanced_mouser_similarity' and mouser_similar.get('eccn_distribution'):
            result['enhanced_details'] = {
                'eccn_distribution': mouser_similar.get('eccn_distribution'),
                'technical_analysis': mouser_similar.get('technical_analysis'),
                'enhanced_confidence': mouser_similar.get('confidence')
            }
        
        if debug:
            result['debug_info'] = {
                'mouser_similar_details': mouser_similar,
                'websearch_details': websearch,
                'llm_classification_details': classification,
                'pdf_evidence': classification.get('pdf_evidence', []),
                'decision_factors': classification.get('decision_factors', [])
            }
        
        return result
    
    def _get_pdf_content(self, s3_key: str) -> Optional[str]:
        """從S3獲取PDF內容"""
        try:
            self.logger.info(f"📄 從S3獲取PDF內容: {s3_key}")
            
            response = self.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
            content = response['Body'].read().decode('utf-8')
            
            self.logger.info(f"✅ 成功獲取PDF內容 ({len(content)} 字符)")
            return content
            
        except Exception as e:
            self.logger.error(f"❌ 無法獲取PDF內容: {str(e)}")
            return None
    
    def _create_success_response(self, data: Dict) -> Dict:
        """創建成功回應"""
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': {
                'success': True,
                'classification': data
            }
        }
    
    def _create_error_response(self, message: str, s3_key: str = None) -> Dict:
        """創建錯誤回應"""
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': {
                'success': False,
                'error': message,
                's3_key': s3_key,
                'timestamp': datetime.now().isoformat()
            }
        }

def lambda_handler(event, context):
    """Lambda處理函數"""
    classifier = CompletePipelineECCNClassifier()
    
    try:
        # 解析請求
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', event)
        
        s3_key = body.get('s3_key')
        product_model = body.get('product_model')
        debug = body.get('debug', False)
        
        if not s3_key or not product_model:
            return classifier._create_error_response("缺少必要參數: s3_key 和 product_model")
        
        # 執行完整Pipeline分類
        return classifier.classify_eccn(s3_key, product_model, debug)
        
    except Exception as e:
        return classifier._create_error_response(f"請求處理失敗: {str(e)}")

# 本地測試
if __name__ == "__main__":
    # 模擬Lambda事件
    test_event = {
        'body': {
            's3_key': 'parsed/pdf_20250728_033546_591daade.json',
            'product_model': 'EKI-5729FI-MB',
            'debug': True
        }
    }
    
    print("🧪 測試完整Pipeline ECCN分類器...")
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2, ensure_ascii=False))