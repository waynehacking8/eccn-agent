#!/usr/bin/env python3
"""
ECCN分類工具增強系統
整合外部API和WebSearch功能以提高分類準確性
"""

import json
import requests
import re
import time
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

class ECCNToolEnhancer:
    """ECCN分類工具增強器"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # API配置 - 從環境變量讀取
        import os
        self.mouser_api_key = os.environ.get('MOUSER_API_KEY', "773b916e-0f6c-4a86-a896-b3b435be5389")  # Mouser API密鑰
        self.digikey_api_key = None  # 需要註冊獲取
        
        # 工具配置
        self.tools_config = {
            'websearch': {'enabled': True, 'timeout': 30},
            'mouser_api': {'enabled': True, 'timeout': 15},
            'digikey_api': {'enabled': False, 'timeout': 15},  # 可選
            'cross_reference': {'enabled': True, 'timeout': 45}
        }
        
        # ECCN模式和關鍵字庫
        self.eccn_patterns = {
            'EAR99': {
                'keywords': ['commercial', 'consumer', 'office', 'unmanaged', 'basic'],
                'temp_range': [(0, 70), (-20, 70)],
                'features': ['simple', 'standard', 'basic']
            },
            '5A991': {
                'keywords': ['industrial', 'ethernet', 'switch', 'managed'],
                'temp_range': [(-40, 85), (-20, 80)],
                'features': ['managed', 'industrial', 'rugged']
            },
            '5A991.b': {
                'keywords': ['security', 'encryption', 'vlan', 'advanced'],
                'temp_range': [(-40, 85)],
                'features': ['security', 'encryption', 'advanced_management']
            },
            '5A991.b.1': {
                'keywords': ['high-speed', 'gigabit', 'fiber', 'performance'],
                'temp_range': [(-40, 85)],
                'features': ['high_speed', 'gigabit', 'fiber_optic']
            },
            '4A994': {
                'keywords': ['management', 'monitoring', 'control', 'poe'],
                'temp_range': [(-40, 85)],
                'features': ['power_management', 'monitoring', 'control_functions']
            }
        }

    def search_mouser_eccn(self, product_model: str) -> Optional[Dict]:
        """
        使用Mouser API查詢產品的官方ECCN分類
        """
        try:
            self.logger.info(f" Mouser API查詢: {product_model}")
            
            # Mouser API搜尋端點 (修正為v1.0格式)
            url = f"https://api.mouser.com/api/v1.0/search/keyword?apiKey={self.mouser_api_key}"
            
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            # 嘗試多種搜尋策略
            search_strategies = [
                product_model,  # 基本搜尋
                f"Advantech {product_model}",  # 包含製造商
                product_model.replace('-', ''),  # 移除連字符
                f"{product_model}-02A1E",  # 常見變體
                f"577-{product_model}",  # Mouser 編號格式
                product_model.split('-')[0] if '-' in product_model else product_model,  # 基本型號
            ]
            
            for search_keyword in search_strategies:
                self.logger.info(f"🔍 嘗試搜尋: {search_keyword}")
                
                payload = {
                    "SearchByKeywordRequest": {
                        "keyword": search_keyword,
                        "records": 10,  # 增加搜尋結果
                        "startingRecord": 0,
                        "searchOptions": "ExcludeNonBuyable",
                        "searchWithYourSignUpLanguage": "false"
                    }
                }
                
                response = requests.post(
                    url, 
                    headers=headers, 
                    json=payload,
                    timeout=self.tools_config['mouser_api']['timeout']
                )
                
                if response.status_code != 200:
                    self.logger.warning(f"⚠️ Mouser API請求失敗: {response.status_code} - {search_keyword}")
                    continue
                    
                try:
                    data = response.json()
                    parts = data.get('SearchResults', {}).get('Parts', [])
                    self.logger.info(f"📊 Mouser API響應: {search_keyword} - {len(parts)} 個結果")
                except Exception as json_error:
                    self.logger.error(f"❌ Mouser API JSON解析失敗: {search_keyword} - {str(json_error)}")
                    continue
                
                self.logger.info(f"📊 搜尋 '{search_keyword}' 找到 {len(parts)} 個產品")
                
                # 檢查每個找到的產品
                for part in parts:
                    part_number = part.get('MouserPartNumber', '')
                    manufacturer = part.get('Manufacturer', '')
                    description = part.get('Description', '')
                    
                    # 調試：顯示產品的所有ECCN相關欄位
                    compliance = part.get('ProductCompliance', [])
                    legacy_eccn = part.get('ExportControlClassificationNumber', '')
                    
                    self.logger.info(f"🔍 檢查產品: {part_number} ({manufacturer})")
                    self.logger.info(f"📋 ProductCompliance: {compliance}")
                    self.logger.info(f"📋 Legacy ECCN: '{legacy_eccn}'")
                    
                    # 使用改進的匹配邏輯
                    if self._is_related_product(part_number, description, product_model, manufacturer):
                        self.logger.info(f"🔍 初步匹配產品: {part_number}")
                        
                        # **新增規格校對步驟**
                        if not self._verify_product_specifications(part, product_model, description):
                            self.logger.info(f"⚠️ 規格校對不符，跳過: {part_number}")
                            continue
                            
                        self.logger.info(f"✅ 規格校對通過: {part_number}")
                        
                        # 1. 檢查ProductCompliance中的ECCN (優先)
                        compliance = part.get('ProductCompliance', [])
                        for comp in compliance:
                            if comp.get('ComplianceName') == 'ECCN':
                                eccn = comp.get('ComplianceValue', '').strip()
                                # 更嚴格的ECCN驗證 - 必須是有效的ECCN格式
                                if eccn and eccn not in ['N/A', 'Not Available', '', 'TBD', 'null', 'NULL', '-', '—'] and self._is_valid_eccn_format(eccn):
                                    eccn = self._normalize_eccn_format(eccn)
                                    self.logger.info(f"✅ Mouser找到有效ECCN: {eccn} (搜尋詞: {search_keyword})")
                                    return {
                                        'source': 'mouser_api_direct',
                                        'eccn_code': eccn,
                                        'part_number': part_number,
                                        'manufacturer': manufacturer,
                                        'description': description,
                                        'confidence': 'high',
                                        'method': 'product_compliance',
                                        'search_keyword': search_keyword,
                                        'reasoning': f'Mouser API直接查詢 "{search_keyword}" 找到產品 {part_number}: {eccn}'
                                    }
                                else:
                                    self.logger.info(f"❌ Mouser ECCN無效或為空: '{eccn}' (產品: {part_number})")
                        
                        # 2. 備用：檢查舊格式的ECCN欄位
                        eccn = part.get('ExportControlClassificationNumber', '').strip()
                        if eccn and eccn not in ['N/A', 'Not Available', '', 'TBD', 'null', 'NULL', '-', '—'] and self._is_valid_eccn_format(eccn):
                            eccn = self._normalize_eccn_format(eccn)
                            self.logger.info(f"✅ Mouser找到有效ECCN (Legacy): {eccn} (搜尋詞: {search_keyword})")
                            return {
                                'source': 'mouser_api_direct',
                                'eccn_code': eccn,
                                'part_number': part_number,
                                'manufacturer': manufacturer,
                                'description': description,
                                'confidence': 'high',
                                'method': 'legacy_field',
                                'search_keyword': search_keyword,
                                'reasoning': f'Mouser API直接查詢 "{search_keyword}" 找到產品 {part_number}: {eccn}'
                            }
                        else:
                            self.logger.info(f"❌ Mouser Legacy ECCN無效或為空: '{eccn}' (產品: {part_number})")
            
            # 所有搜尋策略都沒找到
            self.logger.warning(f"⚠️ Mouser所有搜尋策略都未找到ECCN: {search_strategies}")
            return None
                
        except Exception as e:
            self.logger.error(f" Mouser API例外: {str(e)}")
            return None

    def search_web_eccn_references(self, product_model: str, product_description: str = "") -> List[Dict]:
        """
        使用WebSearch查詢ECCN相關資訊
        """
        try:
            self.logger.info(f" WebSearch查詢: {product_model}")
            
            # 構建搜索查詢
            search_queries = [
                f"{product_model} ECCN export control classification",
                f"{product_model} export control number",
                f'"{product_model}" ECCN',
                f"{product_model} commerce control list"
            ]
            
            results = []
            
            for query in search_queries:
                try:
                    # 模擬WebSearch請求 (需要實際API實現)
                    search_result = self._perform_web_search(query)
                    if search_result:
                        results.extend(search_result)
                        
                    time.sleep(1)  # 避免過度請求
                except Exception as e:
                    self.logger.warning(f"搜索查詢失敗: {query} - {str(e)}")
                    continue
            
            # 分析搜索結果中的ECCN模式
            eccn_matches = self._extract_eccn_from_search_results(results)
            
            self.logger.info(f" WebSearch找到{len(eccn_matches)}個ECCN參考")
            return eccn_matches
            
        except Exception as e:
            self.logger.error(f" WebSearch例外: {str(e)}")
            return []

    def _perform_web_search(self, query: str) -> List[Dict]:
        """
        執行實際的網路搜索 (需要搜索API)
        這裡使用模擬實現，實際需要整合真實搜索API
        """
        # 模擬搜索結果
        mock_results = [
            {
                'title': f'ECCN Classification for {query.split()[0]}',
                'url': 'https://example.com/eccn-database',
                'snippet': 'This product is classified under ECCN 5A991 according to US export control regulations...',
                'source': 'regulatory_database'
            }
        ]
        return mock_results

    def _extract_eccn_from_search_results(self, search_results: List[Dict]) -> List[Dict]:
        """從搜索結果中提取ECCN代碼"""
        eccn_pattern = re.compile(r'\b(EAR99|[0-9][A-Z][0-9]{3}(?:\.[a-z](?:\.[0-9]+)?)?)\b')
        
        matches = []
        for result in search_results:
            text = f"{result.get('title', '')} {result.get('snippet', '')}"
            found_eccns = eccn_pattern.findall(text)
            
            for eccn in found_eccns:
                matches.append({
                    'source': 'websearch',
                    'eccn_code': eccn,
                    'url': result.get('url'),
                    'context': result.get('snippet', ''),
                    'confidence': self._assess_search_confidence(result, eccn)
                })
        
        return matches

    def _assess_search_confidence(self, result: Dict, eccn: str) -> str:
        """評估搜索結果的可信度"""
        confidence_indicators = {
            'high': ['bis.doc.gov', 'export.gov', 'official', 'government'],
            'medium': ['manufacturer', 'datasheet', 'specification'],
            'low': ['forum', 'discussion', 'estimate', 'approximate']
        }
        
        url = result.get('url', '').lower()
        text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        
        for level, indicators in confidence_indicators.items():
            if any(indicator in url or indicator in text for indicator in indicators):
                return level
        
        return 'low'

    def cross_reference_eccn(self, product_model: str, pdf_content: str = "") -> Dict:
        """
        交叉參考多個來源的ECCN資訊
        """
        self.logger.info(f" 開始交叉參考: {product_model}")
        
        sources = {}
        
        # 1. Mouser API查詢
        if self.tools_config['mouser_api']['enabled']:
            mouser_result = self.search_mouser_eccn(product_model)
            if mouser_result:
                sources['mouser'] = mouser_result
        
        # 2. 增強型WebSearch查詢
        if self.tools_config['websearch']['enabled']:
            try:
                from websearch import ECCNWebSearcher
                web_searcher = ECCNWebSearcher(self.logger)
                web_results = web_searcher.search_eccn_information(product_model, "Advantech")
                
                if web_results:
                    sources['websearch'] = web_results[:5]  # 限制數量
                    self.logger.info(f"✅ WebSearch找到 {len(web_results)} 個結果")
                else:
                    # 備用：使用原始搜索
                    web_results = self.search_web_eccn_references(product_model, pdf_content)
                    if web_results:
                        sources['websearch'] = web_results
            except Exception as e:
                self.logger.warning(f"⚠️ 增強型WebSearch失敗，使用備用搜索: {str(e)}")
                # 備用：使用原始搜索
                web_results = self.search_web_eccn_references(product_model, pdf_content)
                if web_results:
                    sources['websearch'] = web_results
        
        # 3. 技術分析 (基於PDF內容)
        if pdf_content:
            tech_analysis = self._analyze_technical_features(pdf_content)
            sources['technical_analysis'] = tech_analysis
        
        # 4. 綜合分析
        final_recommendation = self._synthesize_eccn_sources(sources, product_model)
        
        self.logger.info(f" 交叉參考完成: {final_recommendation.get('eccn_code', 'Unknown')}")
        return final_recommendation

    def _analyze_technical_features(self, pdf_content: str) -> Dict:
        """基於PDF技術內容分析ECCN"""
        features_found = {}
        
        # 分析技術特徵
        for eccn, patterns in self.eccn_patterns.items():
            score = 0
            found_features = []
            
            # 關鍵字匹配
            for keyword in patterns['keywords']:
                if keyword.lower() in pdf_content.lower():
                    score += 1
                    found_features.append(keyword)
            
            # 溫度範圍分析
            temp_matches = re.findall(r'(-?\d+)\s*[°℃C]\s*(?:to|~|-)\s*([+-]?\d+)\s*[°℃C]', pdf_content)
            for temp_range in patterns.get('temp_range', []):
                for match in temp_matches:
                    temp_min, temp_max = int(match[0]), int(match[1])
                    if temp_min >= temp_range[0] and temp_max <= temp_range[1]:
                        score += 2
                        found_features.append(f"temp_range_{temp_min}_{temp_max}")
            
            if score > 0:
                features_found[eccn] = {
                    'score': score,
                    'features': found_features,
                    'confidence': 'high' if score >= 3 else 'medium' if score >= 2 else 'low'
                }
        
        # 選擇最高分的ECCN
        if features_found:
            best_eccn = max(features_found.keys(), key=lambda x: features_found[x]['score'])
            return {
                'source': 'technical_analysis',
                'eccn_code': best_eccn,
                'confidence': features_found[best_eccn]['confidence'],
                'features_found': features_found[best_eccn]['features'],
                'analysis_details': features_found
            }
        
        return {
            'source': 'technical_analysis',
            'eccn_code': 'EAR99',  # 預設為商用
            'confidence': 'low',
            'features_found': [],
            'reason': 'insufficient_technical_features'
        }

    def _synthesize_eccn_sources(self, sources: Dict, product_model: str) -> Dict:
        """綜合多個來源的ECCN資訊"""
        recommendations = []
        
        # 收集所有來源的建議
        for source_name, source_data in sources.items():
            if source_name == 'websearch' and isinstance(source_data, list):
                for item in source_data:
                    recommendations.append(item)
            elif isinstance(source_data, dict) and 'eccn_code' in source_data:
                recommendations.append(source_data)
        
        if not recommendations:
            return {
                'eccn_code': 'EAR99',
                'confidence': 'low',
                'method': 'default_fallback',
                'reasoning': '無外部資料源確認，預設為商用分類',
                'sources_consulted': list(sources.keys())
            }
        
        # 信心度權重
        confidence_weights = {'high': 3, 'medium': 2, 'low': 1}
        
        # 來源權重
        source_weights = {
            'mouser': 5,
            'digikey': 5,
            'technical_analysis': 3,
            'websearch': 2
        }
        
        # 計算加權分數
        eccn_scores = {}
        for rec in recommendations:
            eccn = rec.get('eccn_code', 'EAR99')
            confidence = rec.get('confidence', 'low')
            source = rec.get('source', 'unknown')
            
            score = confidence_weights.get(confidence, 1) * source_weights.get(source, 1)
            
            if eccn not in eccn_scores:
                eccn_scores[eccn] = {'score': 0, 'sources': [], 'details': []}
            
            eccn_scores[eccn]['score'] += score
            eccn_scores[eccn]['sources'].append(source)
            eccn_scores[eccn]['details'].append(rec)
        
        # 選擇最高分的ECCN
        if eccn_scores:
            best_eccn = max(eccn_scores.keys(), key=lambda x: eccn_scores[x]['score'])
            best_data = eccn_scores[best_eccn]
            
            # 計算綜合信心度
            total_score = best_data['score']
            max_possible = len(recommendations) * 3 * 5  # 最高可能分數
            confidence_ratio = total_score / max_possible if max_possible > 0 else 0
            
            if confidence_ratio >= 0.7:
                final_confidence = 'high'
            elif confidence_ratio >= 0.4:
                final_confidence = 'medium'
            else:
                final_confidence = 'low'
            
            return {
                'eccn_code': best_eccn,
                'confidence': final_confidence,
                'method': 'cross_reference_synthesis',
                'reasoning': f'基於{len(best_data["sources"])}個來源的交叉驗證: {", ".join(set(best_data["sources"]))}',
                'score_details': eccn_scores,
                'sources_consulted': list(sources.keys()),
                'total_sources': len(recommendations)
            }
        
        # 回退到預設
        return {
            'eccn_code': 'EAR99',
            'confidence': 'low',
            'method': 'synthesis_fallback',
            'reasoning': '綜合分析後無法確定，預設為商用分類',
            'sources_consulted': list(sources.keys())
        }

    def _is_model_match(self, api_model: str, target_model: str) -> bool:
        """檢查API返回的型號是否匹配目標型號"""
        # 移除常見的分隔符和變體
        api_clean = re.sub(r'[-_\s]', '', api_model.upper())
        target_clean = re.sub(r'[-_\s]', '', target_model.upper())
        
        return api_clean == target_clean or target_clean in api_clean

    def _is_related_product(self, part_number: str, description: str, target_model: str, manufacturer: str, strict_match: bool = True) -> bool:
        """改進的產品匹配邏輯，檢查產品是否相關
        
        Args:
            strict_match: True for exact matches (direct API), False for fuzzy matches (similarity search)
        """
        if not part_number:
            return False
        
        # 製造商必須是Advantech（針對WISE系列）
        if target_model.startswith('WISE') and manufacturer.lower() != 'advantech':
            return False
        
        target_clean = target_model.upper().replace('-', '').replace('_', '')
        part_clean = part_number.upper().replace('-', '').replace('_', '')
        
        if strict_match:
            # 寬鬆名稱匹配 + 嚴格規格校對策略
            # 允許更多候選產品進入，由規格校對來精確識別
            if target_clean == part_clean:
                return True
            # 允許目標型號包含在部件號中 (恢復原來的寬鬆匹配)
            if target_clean in part_clean:
                return True
            return False
        else:
            # 模糊匹配：用於相似產品搜索
            if target_clean in part_clean:
                return True
            
            # 描述匹配：產品描述中包含目標型號
            if description and target_model.upper() in description.upper():
                return True
        
        return False
    
    def _verify_product_specifications(self, mouser_part: dict, target_model: str, pdf_content: str = "") -> bool:
        """
        規格校對：確認Mouser產品與PDF產品規格一致 (寬鬆版本)
        只校對關鍵的不匹配情況
        """
        try:
            part_number = mouser_part.get('MouserPartNumber', '')
            description = mouser_part.get('Description', '').lower()
            manufacturer = mouser_part.get('Manufacturer', '').lower()
            
            # 基本檢查：製造商必須是Advantech
            if 'advantech' not in manufacturer:
                self.logger.info(f"❌ 製造商不匹配: {manufacturer}")
                return False
            
            # **精確產品匹配** - 避免EKI-5728匹配到EKI-5728I-AE
            target_clean = target_model.upper().replace('-', '').replace('_', '')
            part_clean = part_number.upper().replace('-', '').replace('_', '')
            
            # 移除Mouser產品號前綴 (如 923-)
            import re
            part_core = re.sub(r'^\d+', '', part_clean)
            
            # 精確匹配：必須完全相同
            if target_clean == part_core:
                self.logger.info(f"✅ 精確產品匹配: {target_model} = {part_number}")
                return True
            elif part_core.startswith(target_clean) and len(part_core) > len(target_clean):
                # **功能性差異檢測** - 檢測I、MI、LI等功能代碼差異
                suffix = part_core[len(target_clean):]
                self.logger.info(f"🔍 發現可能變體: {target_model} → {part_number} (後綴: {suffix})")
                
                # 檢測功能性後綴 (I=Industrial, MI=Managed Industrial, LI=Layer2 Industrial等)
                functional_suffixes = ['I', 'MI', 'LI', 'SI', 'FI', 'GI', 'CI', 'PI', 'S', 'M', 'G', 'F', 'C', 'P']
                geographic_suffixes = ['AU', 'US', 'EU', 'AS', 'CN', 'JP', 'KR', 'TW', 'HK', 'UK', 'DE', 'FR', 'U', 'E', 'J', 'K', 'H']
                
                # 首先檢查地理區域差異 (AU, US等)
                for geo_suffix in geographic_suffixes:
                    if suffix == geo_suffix or suffix.startswith(geo_suffix + '-') or suffix.endswith('-' + geo_suffix):
                        self.logger.info(f"❌ 檢測到地理區域差異: {target_model} vs {part_number}")
                        self.logger.info(f"   - 目標產品: {target_model}")
                        self.logger.info(f"   - Mouser產品: {part_number} (區域: {geo_suffix})")
                        self.logger.info(f"   - 後綴分析: '{suffix}' 匹配區域代碼 '{geo_suffix}'")
                        return False
                
                # 再檢查功能性差異
                for func_suffix in functional_suffixes:
                    if suffix.startswith(func_suffix):
                        self.logger.info(f"❌ 檢測到功能性差異: {target_model} vs {part_number}")
                        self.logger.info(f"   - 目標產品: {target_model} (基本型)")
                        self.logger.info(f"   - Mouser產品: {part_number} (功能: {func_suffix})")
                        return False
                
                # 非功能性後綴，允許通過
                self.logger.info(f"✅ 允許變體進入規格校對: {part_number}")
                return True
            else:
                self.logger.info(f"❌ 產品型號不匹配: {target_model} vs {part_number}")
            
            # 系列檢查
            target_series = self._extract_product_series(target_model)
            part_series = self._extract_product_series(part_number)
            
            if target_series and part_series and target_series != part_series:
                if not (target_series in part_series or part_series in target_series):
                    self.logger.info(f"❌ 產品系列差異過大: {target_series} vs {part_series}")
                    return False
                
            self.logger.info(f"✅ 規格校對通過: {part_number}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ 規格校對異常，允許通過: {str(e)}")
            return True  # 異常時允許通過
    
    def _extract_product_series(self, model: str) -> str:
        """提取產品系列代碼 - 支持 Mouser 格式"""
        if not model:
            return ""
        
        import re
        
        # 移除 Mouser 產品號前綴 (如 923-)
        clean_model = re.sub(r'^\d+-', '', model.upper())
        
        # 提取主要產品系列
        # EKI-2525LI-AE -> EKI-2525LI (保留功能代碼)
        # EKI-2705G-1GPI-A -> EKI-2705G-1GPI (去掉地區代碼)
        
        # 先匹配複雜格式 (包含連字符和功能代碼)
        match = re.match(r'(EKI-\d{4}[A-Z]*(?:-\d*[A-Z]*)*)', clean_model)
        if match:
            base_series = match.group(1)
            # 去掉最後的地區代碼
            base_series = re.sub(r'-([A-Z]{1,3})$', '', base_series)
            return base_series
            
        # 簡單格式匹配
        match = re.match(r'(EKI-\d{4}[A-Z]*)', clean_model)
        if match:
            return match.group(1)
            
        return clean_model[:15] if len(clean_model) >= 15 else clean_model
    
    def _region_codes_compatible(self, target_model: str, part_number: str) -> bool:
        """檢查地區代碼是否兼容 - 防止 A vs AU 的錯誤匹配"""
        import re
        
        # 提取地區代碼
        target_region = self._extract_region_code(target_model)
        part_region = self._extract_region_code(part_number)
        
        # 完全匹配是最理想的
        if target_region == part_region:
            self.logger.info(f"✅ 地區代碼完全匹配: {target_region}")
            return True
        
        # 檢查是否是不兼容的地區代碼 - 更智能的檢查
        if target_region and part_region and target_region != part_region:
            # 特別檢查明顯衝突的情況
            conflicting_pairs = [
                ('A', 'AU'), ('AU', 'A'),  # A vs AU 衝突
                ('AE', 'BE'), ('BE', 'AE'), # 不同地區版本
                ('US', 'EU'), ('EU', 'US')  # 不同市場版本
            ]
            
            for pair in conflicting_pairs:
                if (target_region, part_region) == pair:
                    self.logger.info(f"❌ 地區代碼衝突: {target_region} vs {part_region}")
                    return False
            
            # 對於其他地區代碼差異，給予更多寬容
            # 例如 -AE vs -02A1E 可能是同一產品的不同版本
            if len(target_region) <= 2 and len(part_region) > 3:
                self.logger.info(f"ℹ️ 地區代碼版本差異: {target_region} vs {part_region} (允許)")
                return True
            elif len(part_region) <= 2 and len(target_region) > 3:
                self.logger.info(f"ℹ️ 地區代碼版本差異: {target_region} vs {part_region} (允許)")
                return True
            
            # 其他不明確的差異給予警告但允許通過
            self.logger.info(f"⚠️ 地區代碼差異: {target_region} vs {part_region} (允許但需注意)")
            return True
        
        # 如果一個有地區代碼，一個沒有，允許通過
        self.logger.info(f"ℹ️ 地區代碼差異: {target_region} vs {part_region} (允許)")
        return True
    
    def _extract_region_code(self, model: str) -> str:
        """提取地區代碼"""
        import re
        
        # 匹配最後的地區代碼 (如 -A, -AU, -BE, -AE 等)
        match = re.search(r'-([A-Z]{1,3})$', model.upper())
        if match:
            return match.group(1)
        return ""
    
    def _has_connector_mismatch(self, target_model: str, part_number: str, description: str) -> bool:
        """檢查連接器類型是否匹配"""
        target_upper = target_model.upper()
        part_upper = part_number.upper()
        desc_lower = description.lower()
        
        # M12連接器檢查
        target_has_m12 = 'M12' in target_upper
        part_has_m12 = 'M12' in part_upper or 'm12' in desc_lower
        
        if target_has_m12 != part_has_m12:
            self.logger.info(f"⚠️ M12連接器差異: target={target_has_m12}, part={part_has_m12} (允許)")
            # 改為警告而非拒絕，因為描述可能不完整
            # return True 表示不拒絕
        
        # 光纖連接器檢查 (F, FI, LX, SX)
        fiber_indicators = ['F-', 'FI-', 'LX-', 'SX-', 'G-', 'GI-']
        target_has_fiber = any(ind in target_upper for ind in fiber_indicators)
        part_has_fiber = any(ind in part_upper for ind in fiber_indicators) or 'fiber' in desc_lower
        
        if target_has_fiber != part_has_fiber:
            self.logger.info(f"⚠️ 光纖連接器差異: target={target_has_fiber}, part={part_has_fiber} (允許)")
            # 改為警告而非拒絕
            
        return False
    
    def _has_function_mismatch(self, target_model: str, part_number: str, description: str) -> bool:
        """檢查功能類型是否匹配"""
        target_upper = target_model.upper()
        part_upper = part_number.upper()
        desc_lower = description.lower()
        
        # GPI/GFPI 功能檢查 (PoE注入器)
        target_is_gpi = any(gpi in target_upper for gpi in ['GPI', 'GFPI'])
        part_is_gpi = any(gpi in part_upper for gpi in ['GPI', 'GFPI']) or 'injector' in desc_lower
        
        if target_is_gpi != part_is_gpi:
            # GPI功能差異比較重要，因為這影響產品類型 (PoE注入器 vs 交換機)
            self.logger.info(f"❌ GPI功能不匹配: target={target_is_gpi}, part={part_is_gpi}")
            return True
        
        # 管理功能檢查 (M vs 非M)
        target_is_managed = '-M-' in target_upper or target_upper.endswith('-M')
        part_is_managed = '-M-' in part_upper or part_upper.endswith('-M') or 'managed' in desc_lower
        
        # 對於管理功能，允許一定靈活性，但記錄差異
        if target_is_managed != part_is_managed:
            self.logger.info(f"ℹ️ 管理功能差異: target={target_is_managed}, part={part_is_managed}")
            # 不作為否決條件，因為描述可能不完整
        
        return False
    
    def _is_valid_eccn_format(self, eccn_code: str) -> bool:
        """驗證ECCN代碼格式是否有效"""
        if not eccn_code or not isinstance(eccn_code, str):
            return False
            
        eccn_code = eccn_code.strip().upper()
        
        # EAR99 是有效格式
        if eccn_code == 'EAR99':
            return True
            
        # 標準ECCN格式：數字+字母+3位數字，可選子分類
        # 例如：5A991, 5A991.b, 5A991.b.1, 4A994
        import re
        pattern = r'^[0-9][A-Z][0-9]{3}(?:\.[a-zA-Z](?:\.[0-9]+)?)?$'
        return bool(re.match(pattern, eccn_code))
    
    def _normalize_eccn_format(self, eccn_code: str) -> str:
        """標準化ECCN格式：子分類字母轉為小寫"""
        # EAR99 保持不變
        if eccn_code == 'EAR99':
            return eccn_code
            
        # 對於類似 5A992.C 的格式，將子分類字母轉為小寫
        pattern = r'(\d[A-Z]\d{3})\.([A-Z])(\.\d+)?'
        match = re.match(pattern, eccn_code)
        if match:
            base = match.group(1)  # 例如 5A992
            subcategory = match.group(2).lower()  # C -> c
            version = match.group(3) if match.group(3) else ''  # .1 或空
            return f"{base}.{subcategory}{version}"
        
        # 如果格式不匹配，返回原始值
        return eccn_code

    def enhance_classification(self, product_model: str, pdf_content: str, 
                             original_classification: Dict) -> Dict:
        """
        增強原始分類結果
        """
        self.logger.info(f" 開始工具增強分類: {product_model}")
        
        # 執行交叉參考
        cross_ref_result = self.cross_reference_eccn(product_model, pdf_content)
        
        # 比較原始分類和工具增強結果
        original_eccn = original_classification.get('eccn_code', 'Unknown')
        external_eccn = cross_ref_result.get('eccn_code', 'Unknown')
        
        # 決定最終分類
        if cross_ref_result.get('confidence') == 'high' and external_eccn != original_eccn:
            # 高信心度的外部來源優先
            final_classification = cross_ref_result
            final_classification['validation_decision'] = 'external_override'
            final_classification['original_classification'] = original_classification
        elif cross_ref_result.get('confidence') in ['medium', 'high'] and external_eccn == original_eccn:
            # 外部來源確認原始分類
            final_classification = original_classification.copy()
            final_classification['confidence'] = 'high'  # 提升信心度
            final_classification['validation_decision'] = 'external_confirmation'
            final_classification['cross_reference_result'] = cross_ref_result
        else:
            # 保持原始分類但記錄不一致
            final_classification = original_classification.copy()
            final_classification['validation_decision'] = 'original_maintained'
            final_classification['cross_reference_result'] = cross_ref_result
            final_classification['consistency_note'] = f'外部來源建議{external_eccn}，但保持原始分類{original_eccn}'
        
        final_classification['tool_validation'] = True
        final_classification['validation_timestamp'] = datetime.now().isoformat()
        
        self.logger.info(f"✅ 工具驗證完成: {final_classification.get('eccn_code')} (決策: {final_classification.get('validation_decision')})")
        
        return final_classification

# 使用示例
def example_usage():
    """使用示例"""
    import logging
    
    # 設置日誌
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 創建工具增強器
    enhancer = ECCNToolEnhancer(logger)
    
    # 模擬原始分類結果
    original_result = {
        'eccn_code': '5A991',
        'confidence': 'medium',
        'method': 'ai_classification',
        'reasoning': 'Industrial Ethernet switch with managed features'
    }
    
    # 模擬PDF內容
    pdf_content = """
    EKI-2528G Industrial Ethernet Switch
    24 Gigabit Ethernet ports + 4 Combo ports
    Operating Temperature: -40°C to 75°C
    Managed switch with VLAN support
    Industrial grade housing
    """
    
    # 執行工具增強
    validated_result = enhancer.enhance_classification(
        product_model="EKI-2528G",
        pdf_content=pdf_content,
        original_classification=original_result
    )
    
    print(" 工具增強結果:")
    print(json.dumps(validated_result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    example_usage()#!/usr/bin/env python3
"""
Mouser API 整合模組
提供真實的 Mouser Electronics API 整合功能
用於查詢產品的官方 ECCN 分類資訊
"""

import json
import requests
import re
import time
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

class MouserAPIClient:
    """Mouser Electronics API 客戶端 with Feature-Based Search"""
    
    def __init__(self, api_key: str = None, logger: logging.Logger = None):
        self.api_key = api_key or "d337cdb2-f839-4405-b34a-2533df7c60af"
        self.logger = logger or logging.getLogger(__name__)
        
        # Mouser API 端點 (使用v1.0)
        self.base_url = "https://api.mouser.com"
        self.search_endpoint = f"{self.base_url}/api/v1.0/search/keyword"
        self.part_detail_endpoint = f"{self.base_url}/api/v1/search/partnumber"
        
        # API 配置
        self.timeout = 30
        self.max_retries = 3
        self.rate_limit_delay = 1  # 秒
        
        # 請求標頭
        self.headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
            
        # ECCN 模式匹配
        self.eccn_pattern = re.compile(r'\b(EAR99|[0-9][A-Z][0-9]{3}(?:\.[a-zA-Z](?:\.[0-9]+)?)?)\b')
        
        # PDF特徵提取相關配置
        self.feature_keywords = {
            'management': ['snmp', 'web', 'gui', 'cli', 'management', 'managed', 'configuration', 'monitoring'],
            'switching': ['switch', 'ethernet', 'network', 'port', 'gigabit', 'switching', 'layer2', 'layer3'],
            'industrial': ['industrial', 'din-rail', 'rugged', 'wide temperature', 'harsh environment'],
            'protocols': ['modbus', 'profinet', 'tcp', 'ip', 'ethernet', 'bacnet', 'opcua'],
            'security': ['encryption', 'vpn', 'firewall', '802.1x', 'authentication', 'security', 'access control'],
            'performance': ['gigabit', '10g', 'switching capacity', 'throughput', 'wire-speed', 'backplane'],
            'quality': ['qos', 'vlan', 'link aggregation', 'lacp', 'spanning tree', 'multicast']
        }
    
    def _normalize_eccn_format(self, eccn_code: str) -> str:
        """標準化ECCN格式：子分類字母轉為小寫"""
        # EAR99 保持不變
        if eccn_code == 'EAR99':
            return eccn_code
            
        # 對於類似 5A992.C 的格式，將子分類字母轉為小寫
        import re
        pattern = r'(\d[A-Z]\d{3})\.([A-Z])(\.\d+)?'
        match = re.match(pattern, eccn_code)
        if match:
            base = match.group(1)  # 例如 5A992
            subcategory = match.group(2).lower()  # C -> c
            version = match.group(3) if match.group(3) else ''  # .1 或空
            return f"{base}.{subcategory}{version}"
        
        # 如果格式不匹配，返回原始值
        return eccn_code

    def search_by_keyword(self, keyword: str, max_results: int = 10) -> List[Dict]:
        """
        使用關鍵字搜索產品 (v1.0 API)
        
        Args:
            keyword: 搜索關鍵字 (產品型號等)
            max_results: 最大結果數量
            
        Returns:
            產品清單
        """
        try:
            self.logger.info(f" Mouser關鍵字搜索: {keyword}")
            
            # 使用查詢參數格式
            url = f"{self.search_endpoint}?apiKey={self.api_key}"
            
            payload = {
                "SearchByKeywordRequest": {
                    "keyword": keyword,
                    "records": min(max_results, 50),  # API限制最多50個結果
                    "startingRecord": 0,
                    "searchOptions": "ExcludeNonBuyable,ExcludeObsolete",
                    "searchWithYourSignUpLanguage": "false"
                }
            }
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                search_results = data.get('SearchResults', {})
                parts = search_results.get('Parts', [])
                
                self.logger.info(f" 找到 {len(parts)} 個產品")
                return parts
            else:
                self.logger.warning(f"API回應錯誤: {response.status_code} - {response.text}")
                return []
            
        except Exception as e:
            self.logger.error(f" 關鍵字搜索失敗: {str(e)}")
            return []

    def search_by_part_number(self, part_number: str) -> Optional[Dict]:
        """
        使用精確零件號搜索 (v1.0 API)
        
        Args:
            part_number: 精確的零件號
            
        Returns:
            產品詳情或None
        """
        try:
            self.logger.info(f" Mouser零件號搜索: {part_number}")
            
            # 使用查詢參數格式
            url = f"{self.part_detail_endpoint}?apiKey={self.api_key}"
            
            payload = {
                "SearchByPartRequest": {
                    "mouserPartNumber": part_number,
                    "partSearchOptions": "ExcludeNonBuyable"
                }
            }
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                search_results = data.get('SearchResults', {})
                parts = search_results.get('Parts', [])
                
                if parts:
                    self.logger.info(f" 找到零件: {parts[0].get('MouserPartNumber', 'N/A')}")
                    return parts[0]
                else:
                    self.logger.info("️ 未找到匹配的零件")
                    return None
            else:
                self.logger.warning(f"API回應錯誤: {response.status_code} - {response.text}")
                return None
            
        except Exception as e:
            self.logger.error(f" 零件號搜索失敗: {str(e)}")
            return None

    def get_eccn_info(self, product_model: str, pdf_content: str = None) -> Optional[Dict]:
        """
        獲取產品的 ECCN 資訊 (支持特徵匹配搜索)
        
        Args:
            product_model: 產品型號
            pdf_content: PDF內容 (可選，用於特徵匹配)
            
        Returns:
            ECCN 資訊字典或None
        """
        try:
            self.logger.info(f" 查詢ECCN: {product_model}")
            
            # 1. 嘗試精確零件號搜索
            exact_result = self.search_by_part_number(product_model)
            if exact_result:
                eccn_info = self._extract_eccn_from_part(exact_result, product_model)
                if eccn_info:
                    eccn_info['search_method'] = 'exact_part_number'
                    return eccn_info
            
            # 2. 關鍵字搜索
            keyword_results = self.search_by_keyword(product_model, max_results=20)
            
            # 尋找最佳匹配
            best_match = self._find_best_match(keyword_results, product_model)
            if best_match:
                eccn_info = self._extract_eccn_from_part(best_match, product_model)
                if eccn_info:
                    eccn_info['search_method'] = 'keyword_search'
                    return eccn_info
            
            # 3. 變體搜索 (移除常見後綴)
            variant_models = self._generate_model_variants(product_model)
            for variant in variant_models:
                variant_results = self.search_by_keyword(variant, max_results=10)
                for result in variant_results:
                    if self._is_model_similar(result.get('MouserPartNumber', ''), product_model):
                        eccn_info = self._extract_eccn_from_part(result, product_model)
                        if eccn_info:
                            eccn_info['search_method'] = 'variant_search'
                            eccn_info['variant_used'] = variant
                            return eccn_info
                time.sleep(self.rate_limit_delay)
            
            # 4. 新增：特徵匹配搜索 (當前述方法都失敗時)
            if pdf_content:
                self.logger.info(" 使用PDF特徵進行相似產品搜索...")
                feature_based_result = self.search_by_features(pdf_content, product_model)
                if feature_based_result:
                    return feature_based_result
            
            self.logger.warning(f"️ 未找到 {product_model} 的ECCN資訊")
            return None
            
        except Exception as e:
            self.logger.error(f" ECCN查詢失敗: {str(e)}")
            return None

    def _make_request(self, url: str, payload: Dict) -> Optional[Dict]:
        """發送API請求並處理重試"""
        
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.debug(f"API請求嘗試 {attempt}/{self.max_retries}")
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # API金鑰檢查
                if response.status_code == 401:
                    self.logger.error(" Mouser API認證失敗 - 請檢查API金鑰")
                    return None
                
                # 速率限制檢查
                if response.status_code == 429:
                    self.logger.warning("️ API速率限制 - 等待重試")
                    time.sleep(self.rate_limit_delay * attempt)
                    continue
                
                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.warning(f"API回應錯誤: {response.status_code} - {response.text}")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"API請求超時 (嘗試 {attempt}/{self.max_retries})")
            except Exception as e:
                self.logger.error(f"API請求例外: {str(e)}")
            
            if attempt < self.max_retries:
                time.sleep(self.rate_limit_delay * attempt)
        
        return None

    def _extract_eccn_from_part(self, part_data: Dict, original_model: str) -> Optional[Dict]:
        """從零件資料中提取ECCN資訊"""
        
        try:
            mouser_part_number = part_data.get('MouserPartNumber', '')
            manufacturer = part_data.get('Manufacturer', '')
            description = part_data.get('Description', '')
            
            # 1. 檢查ProductCompliance中的ECCN (正確的API格式)
            compliance = part_data.get('ProductCompliance', [])
            for comp in compliance:
                if comp.get('ComplianceName') == 'ECCN':
                    eccn_number = comp.get('ComplianceValue', '')
                    if eccn_number and eccn_number not in ['N/A', 'Not Available', '', 'TBD']:
                        # 驗證ECCN格式
                        if self.eccn_pattern.match(eccn_number):
                            # 標準化ECCN格式：子分類字母轉為小寫
                            normalized_eccn = self._normalize_eccn_format(eccn_number)
                            confidence = self._calculate_confidence(part_data, original_model)
                            
                            return {
                                'source': 'mouser_api',
                                'eccn_code': normalized_eccn,
                                'mouser_part_number': mouser_part_number,
                                'manufacturer': manufacturer,
                                'product_description': description,
                                'confidence': confidence,
                                'match_score': self._calculate_match_score(mouser_part_number, original_model),
                                'raw_data': part_data,
                                'timestamp': datetime.now().isoformat()
                            }
            
            # 2. 備用：檢查舊格式的ECCN欄位
            eccn_number = part_data.get('ExportControlClassificationNumber', '')
            eccn_description = part_data.get('ExportControlClassificationNumberDescription', '')
            
            if eccn_number and eccn_number not in ['N/A', 'Not Available', '', 'TBD']:
                if self.eccn_pattern.match(eccn_number):
                    # 標準化ECCN格式
                    normalized_eccn = self._normalize_eccn_format(eccn_number)
                    confidence = self._calculate_confidence(part_data, original_model)
                    
                    return {
                        'source': 'mouser_api',
                        'eccn_code': normalized_eccn,
                        'eccn_description': eccn_description,
                        'mouser_part_number': mouser_part_number,
                        'manufacturer': manufacturer,
                        'product_description': description,
                        'confidence': confidence,
                        'match_score': self._calculate_match_score(mouser_part_number, original_model),
                        'raw_data': part_data,
                        'timestamp': datetime.now().isoformat()
                    }
            
            # 3. 最後嘗試：檢查描述中的ECCN模式
            combined_text = f"{description} {eccn_description}".upper()
            eccn_matches = self.eccn_pattern.findall(combined_text)
            
            if eccn_matches:
                # 標準化ECCN格式
                normalized_eccn = self._normalize_eccn_format(eccn_matches[0])
                return {
                    'source': 'mouser_api_extracted',
                    'eccn_code': normalized_eccn,
                    'mouser_part_number': mouser_part_number,
                    'manufacturer': manufacturer,
                    'product_description': description,
                    'confidence': 'medium',
                    'extraction_method': 'text_pattern',
                    'raw_data': part_data,
                    'timestamp': datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"ECCN提取失敗: {str(e)}")
            return None

    def _find_best_match(self, parts: List[Dict], target_model: str) -> Optional[Dict]:
        """在搜索結果中找到最佳匹配"""
        
        if not parts:
            return None
        
        scored_parts = []
        
        for part in parts:
            mouser_pn = part.get('MouserPartNumber', '')
            manufacturer_pn = part.get('ManufacturerPartNumber', '')
            
            # 計算匹配分數
            score = max(
                self._calculate_match_score(mouser_pn, target_model),
                self._calculate_match_score(manufacturer_pn, target_model)
            )
            
            if score > 0.5:  # 只考慮相關性較高的結果
                scored_parts.append((part, score))
        
        if scored_parts:
            # 按分數排序並返回最佳匹配
            scored_parts.sort(key=lambda x: x[1], reverse=True)
            return scored_parts[0][0]
        
        return None

    def _calculate_match_score(self, api_model: str, target_model: str) -> float:
        """計算型號匹配分數 (0-1)"""
        
        if not api_model or not target_model:
            return 0.0
        
        # 標準化字串
        api_clean = re.sub(r'[-_\s]', '', api_model.upper())
        target_clean = re.sub(r'[-_\s]', '', target_model.upper())
        
        # 精確匹配
        if api_clean == target_clean:
            return 1.0
        
        # 包含匹配
        if target_clean in api_clean or api_clean in target_clean:
            return 0.8
        
        # 字首匹配
        if api_clean.startswith(target_clean) or target_clean.startswith(api_clean):
            return 0.7
        
        # 相似度計算 (簡化版本)
        common_chars = sum(1 for a, b in zip(api_clean, target_clean) if a == b)
        max_len = max(len(api_clean), len(target_clean))
        
        if max_len > 0:
            similarity = common_chars / max_len
            return similarity * 0.6  # 降低權重
        
        return 0.0

    def _calculate_confidence(self, part_data: Dict, original_model: str) -> str:
        """計算ECCN資訊的可信度"""
        
        confidence_score = 0
        
        # ECCN資料完整性
        if part_data.get('ExportControlClassificationNumber'):
            confidence_score += 3
        if part_data.get('ExportControlClassificationNumberDescription'):
            confidence_score += 2
        
        # 型號匹配度
        mouser_pn = part_data.get('MouserPartNumber', '')
        manufacturer_pn = part_data.get('ManufacturerPartNumber', '')
        
        match_score = max(
            self._calculate_match_score(mouser_pn, original_model),
            self._calculate_match_score(manufacturer_pn, original_model)
        )
        
        if match_score >= 0.9:
            confidence_score += 3
        elif match_score >= 0.7:
            confidence_score += 2
        elif match_score >= 0.5:
            confidence_score += 1
        
        # 製造商資訊
        if part_data.get('Manufacturer'):
            confidence_score += 1
        
        # 產品狀態
        if part_data.get('ProductStatus') == 'Active':
            confidence_score += 1
        
        # 轉換為信心等級
        if confidence_score >= 7:
            return 'high'
        elif confidence_score >= 4:
            return 'medium'
        else:
            return 'low'

    def _generate_model_variants(self, model: str) -> List[str]:
        """生成型號變體用於搜索"""
        
        variants = []
        
        # 移除常見後綴
        suffixes_to_remove = ['-AE', '-BE', '-MI', '-ST', '-CA', '/US', '/EU', '/JP']
        
        for suffix in suffixes_to_remove:
            if model.endswith(suffix):
                variants.append(model[:-len(suffix)])
        
        # 移除最後的字元 (可能是版本號)
        if len(model) > 3:
            variants.append(model[:-1])
            variants.append(model[:-2])
        
        # 基礎型號 (移除數字後綴)
        base_match = re.match(r'([A-Z]+-?\d+)', model)
        if base_match:
            variants.append(base_match.group(1))
        
        return list(set(variants))  # 去重

    def _is_model_similar(self, api_model: str, target_model: str) -> bool:
        """檢查型號是否相似"""
        return self._calculate_match_score(api_model, target_model) >= 0.6
    
    def extract_pdf_features(self, pdf_content: str) -> Dict[str, Any]:
        """
        從PDF內容中提取產品特徵用於相似產品搜索
        
        Args:
            pdf_content: PDF文本內容
            
        Returns:
            特徵字典
        """
        try:
            self.logger.info(" 提取PDF產品特徵...")
            
            content_lower = pdf_content.lower()
            features = {
                'feature_scores': {},
                'technical_specs': {},
                'search_keywords': []
            }
            
            # 1. 計算各類特徵的分數
            for category, keywords in self.feature_keywords.items():
                score = 0
                matched_keywords = []
                
                for keyword in keywords:
                    if keyword in content_lower:
                        score += 1
                        matched_keywords.append(keyword)
                
                features['feature_scores'][category] = {
                    'score': score,
                    'matched_keywords': matched_keywords,
                    'normalized_score': score / len(keywords) if keywords else 0
                }
            
            # 2. 提取技術規格
            features['technical_specs'] = self._extract_technical_specs(content_lower)
            
            # 3. 生成搜索關鍵字
            features['search_keywords'] = self._generate_search_keywords(features)
            
            # 4. 確定產品類型
            features['product_type'] = self._determine_product_type(features)
            
            self.logger.info(f" 特徵提取完成，產品類型: {features['product_type']}")
            return features
            
        except Exception as e:
            self.logger.error(f" PDF特徵提取失敗: {str(e)}")
            return {}
    
    def _extract_technical_specs(self, content_lower: str) -> Dict[str, Any]:
        """提取技術規格"""
        specs = {}
        
        # 溫度範圍
        temp_pattern = r'operating temperature[:\s]*(-?\d+)[^\d]*(-?\d+)'
        temp_match = re.search(temp_pattern, content_lower)
        if temp_match:
            specs['temp_min'] = int(temp_match.group(1))
            specs['temp_max'] = int(temp_match.group(2))
        
        # 電源規格
        if 'dc power' in content_lower or '12v' in content_lower or '24v' in content_lower:
            specs['power_type'] = 'dc'
        elif '100-240v' in content_lower or 'ac power' in content_lower:
            specs['power_type'] = 'ac'
        
        # 端口數量
        port_pattern = r'(\d+)[^\d]*port'
        port_matches = re.findall(port_pattern, content_lower)
        if port_matches:
            specs['port_count'] = max(int(p) for p in port_matches)
        
        # 切換容量
        capacity_pattern = r'switching capacity[:\s]*(\d+)[^\d]*gbps'
        capacity_match = re.search(capacity_pattern, content_lower)
        if capacity_match:
            specs['switching_capacity_gbps'] = int(capacity_match.group(1))
        
        return specs
    
    def _generate_search_keywords(self, features: Dict) -> List[str]:
        """基於特徵生成搜索關鍵字"""
        keywords = []
        
        # 基於特徵分數生成關鍵字
        for category, data in features['feature_scores'].items():
            if data['normalized_score'] > 0.3:  # 高分特徵
                keywords.extend(data['matched_keywords'][:3])  # 取前3個關鍵字
        
        # 技術規格關鍵字
        specs = features['technical_specs']
        if 'power_type' in specs:
            keywords.append(f"{specs['power_type']} power")
        
        if 'port_count' in specs:
            keywords.append(f"{specs['port_count']} port")
        
        # 去重並返回前10個最相關的關鍵字
        return list(dict.fromkeys(keywords))[:10]
    
    def _determine_product_type(self, features: Dict) -> str:
        """根據特徵確定產品類型"""
        scores = features['feature_scores']
        
        # 管理型交換機指標
        if (scores.get('management', {}).get('normalized_score', 0) > 0.4 and
            scores.get('switching', {}).get('normalized_score', 0) > 0.3 and
            scores.get('quality', {}).get('normalized_score', 0) > 0.3):
            return 'managed_switch'
        
        # 工業級交換機指標
        elif (scores.get('switching', {}).get('normalized_score', 0) > 0.3 and
              scores.get('industrial', {}).get('normalized_score', 0) > 0.2):
            return 'industrial_switch'
        
        # 基本交換機
        elif scores.get('switching', {}).get('normalized_score', 0) > 0.3:
            return 'basic_switch'
        
        # 網路設備
        elif scores.get('switching', {}).get('normalized_score', 0) > 0.1:
            return 'network_equipment'
        
        return 'unknown'
    
    def search_by_features(self, pdf_content: str, original_model: str) -> Optional[Dict]:
        """
        基於PDF特徵搜索相似產品並聚合ECCN
        
        Args:
            pdf_content: PDF內容
            original_model: 原始產品型號
            
        Returns:
            聚合的ECCN資訊或None
        """
        try:
            # 1. 提取特徵
            features = self.extract_pdf_features(pdf_content)
            if not features:
                return None
            
            # 2. 基於特徵搜索相似產品
            similar_products = self._search_similar_products(features)
            if not similar_products:
                self.logger.info(" 未找到相似產品")
                return None
            
            # 3. 提取ECCN並進行聚合分析
            eccn_analysis = self._analyze_similar_products_eccn(similar_products, features)
            if not eccn_analysis:
                return None
            
            # 4. 生成基於特徵的ECCN建議
            return {
                'source': 'mouser_feature_matching',
                'eccn_code': eccn_analysis['recommended_eccn'],
                'confidence': eccn_analysis['confidence'],
                'reasoning': eccn_analysis['reasoning'],
                'search_method': 'feature_matching',
                'feature_analysis': features,
                'similar_products_count': len(similar_products),
                'eccn_distribution': eccn_analysis['eccn_distribution'],
                'matching_features': eccn_analysis['key_features'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f" 特徵匹配搜索失敗: {str(e)}")
            return None
    
    def _search_similar_products(self, features: Dict) -> List[Dict]:
        """基於特徵搜索相似產品"""
        similar_products = []
        
        try:
            # 使用搜索關鍵字組合搜索
            search_keywords = features['search_keywords']
            product_type = features['product_type']
            
            # 構建搜索查詢
            search_queries = []
            
            if product_type == 'managed_switch':
                search_queries = [
                    'managed ethernet switch',
                    'industrial managed switch',
                    'layer 2 managed switch'
                ]
            elif product_type == 'industrial_switch':
                search_queries = [
                    'industrial ethernet switch',
                    'din rail ethernet switch',
                    'rugged ethernet switch'
                ]
            else:
                search_queries = [
                    'ethernet switch',
                    'network switch',
                    'industrial network'
                ]
            
            # 對每個查詢執行搜索
            for query in search_queries[:2]:  # 限制搜索次數
                self.logger.info(f" 搜索查詢: {query}")
                results = self.search_by_keyword(query, max_results=15)
                
                # 過濾相關產品
                relevant_products = self._filter_relevant_products(results, features)
                similar_products.extend(relevant_products)
                
                time.sleep(self.rate_limit_delay)
            
            # 去重並限制數量
            unique_products = []
            seen_parts = set()
            
            for product in similar_products:
                part_number = product.get('MouserPartNumber', '')
                if part_number and part_number not in seen_parts:
                    seen_parts.add(part_number)
                    unique_products.append(product)
                    
                    if len(unique_products) >= 20:  # 限制最多20個產品
                        break
            
            self.logger.info(f" 找到 {len(unique_products)} 個相似產品")
            return unique_products
            
        except Exception as e:
            self.logger.error(f" 相似產品搜索失敗: {str(e)}")
            return []
    
    def _filter_relevant_products(self, products: List[Dict], features: Dict) -> List[Dict]:
        """過濾相關產品"""
        relevant_products = []
        
        for product in products:
            description = product.get('Description', '').lower()
            manufacturer = product.get('Manufacturer', '').lower()
            
            # 基本相關性檢查
            relevance_score = 0
            
            # 檢查產品描述中的關鍵特徵
            for category, data in features['feature_scores'].items():
                for keyword in data['matched_keywords']:
                    if keyword in description:
                        relevance_score += 1
            
            # 如果相關性分數足夠高，加入候選 (降低門檻)
            if relevance_score >= 1:  # 降低門檻從2到1
                relevant_products.append(product)
        
        return relevant_products
    
    def _analyze_similar_products_eccn(self, products: List[Dict], features: Dict) -> Optional[Dict]:
        """分析相似產品的ECCN分佈並推薦"""
        try:
            eccn_data = []
            
            # 提取每個產品的ECCN
            for product in products:
                eccn_info = self._extract_eccn_from_part(product, 'similar')
                if eccn_info and eccn_info.get('eccn_code'):
                    eccn_data.append({
                        'eccn_code': eccn_info['eccn_code'],
                        'product': product,
                        'confidence': eccn_info.get('confidence', 'medium')
                    })
            
            if not eccn_data:
                return None
            
            # 統計ECCN分佈
            eccn_counts = {}
            for item in eccn_data:
                eccn = item['eccn_code']
                if eccn not in eccn_counts:
                    eccn_counts[eccn] = []
                eccn_counts[eccn].append(item)
            
            # 智能ECCN推薦：結合多數決策和技術複雜度
            eccn_scores = self._calculate_eccn_scores(eccn_counts, features)
            
            # 找出得分最高的ECCN
            recommended_eccn = max(eccn_scores.keys(), key=lambda x: eccn_scores[x]['final_score'])
            eccn_details = eccn_scores[recommended_eccn]
            
            count = eccn_details['count']
            total = len(eccn_data)
            percentage = (count / total) * 100
            
            # 確定信心度（基於最終得分）
            final_score = eccn_details['final_score']
            if final_score >= 80:
                confidence = 'high'
            elif final_score >= 60:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            # 生成推理說明
            spec_bonus = eccn_details.get('specification_bonus', 0)
            feature_match = eccn_details.get('feature_match_score', 0)
            
            if spec_bonus > 0:
                reasoning = f"基於{total}個相似產品分析，{count}個產品({percentage:.1f}%)具有{recommended_eccn}分類，規格接近度評判+{spec_bonus}分(特徵匹配度:{feature_match:.1f})"
            else:
                reasoning = f"基於{total}個相似產品分析，{count}個產品({percentage:.1f}%)具有{recommended_eccn}分類"
            
            # 識別關鍵特徵
            key_features = []
            product_type = features.get('product_type', 'unknown')
            if product_type == 'managed_switch':
                key_features = ['管理功能', '高級特性']
            elif product_type == 'industrial_switch':
                key_features = ['工業級設計', '寬溫範圍']
            
            return {
                'recommended_eccn': recommended_eccn,
                'confidence': confidence,
                'reasoning': reasoning,
                'eccn_distribution': {eccn: len(products) for eccn, products in eccn_counts.items()},
                'key_features': key_features,
                'total_products_analyzed': total,
                'eccn_scores': eccn_scores  # 添加詳細評分信息
            }
            
        except Exception as e:
            self.logger.error(f" ECCN分析失敗: {str(e)}")
            return None

    def _calculate_eccn_scores(self, eccn_counts: Dict[str, List], features: Dict) -> Dict[str, Dict]:
        """
        基於規格接近度計算ECCN評分
        使用技術特徵匹配替代簡單多數決策
        """
        try:
            self.logger.info("🧮 計算基於規格接近度的ECCN評分...")
            
            eccn_scores = {}
            
            # 定義ECCN技術特徵權重表
            eccn_feature_profiles = {
                'EAR99': {
                    'management': 0.1,    # 基本管理功能
                    'security': 0.0,      # 無安全功能
                    'performance': 0.2,   # 基本性能
                    'protocols': 0.1,     # 基本協議
                    'industrial': 0.0,    # 非工業級
                    'quality': 0.1        # 基本QoS
                },
                '5A991': {
                    'management': 0.5,    # 中等管理功能
                    'security': 0.2,      # 基本安全功能
                    'performance': 0.4,   # 中等性能
                    'protocols': 0.4,     # 標準協議支援
                    'industrial': 0.6,    # 工業級設計
                    'quality': 0.3        # 標準QoS
                },
                '5A991.b': {
                    'management': 0.7,    # 高級管理功能
                    'security': 0.8,      # 高級安全功能
                    'performance': 0.6,   # 高性能
                    'protocols': 0.6,     # 高級協議支援
                    'industrial': 0.7,    # 強化工業級
                    'quality': 0.7        # 高級QoS
                },
                '5A991.b.1': {
                    'management': 0.8,    # 企業級管理
                    'security': 0.6,      # 企業安全
                    'performance': 0.9,   # 高速性能
                    'protocols': 0.7,     # 高速協議
                    'industrial': 0.6,    # 適度工業化
                    'quality': 0.8        # 企業QoS
                },
                '4A994': {
                    'management': 0.9,    # 專業管理功能
                    'security': 0.4,      # 中等安全
                    'performance': 0.5,   # 中等性能
                    'protocols': 0.8,     # 管理協議專精
                    'industrial': 0.5,    # 部分工業化
                    'quality': 0.6        # 管理導向QoS
                },
                '5A992.c': {
                    'management': 0.9,    # 綜合管理功能
                    'security': 0.9,      # 最高安全功能
                    'performance': 0.8,   # 高端性能
                    'protocols': 0.9,     # 綜合協議支援
                    'industrial': 0.8,    # 高階工業級
                    'quality': 0.9        # 最高級QoS
                }
            }
            
            # 獲取目標產品的特徵評分
            target_features = features.get('feature_scores', {})
            
            # 對每個ECCN計算評分
            for eccn_code, products_list in eccn_counts.items():
                base_count = len(products_list)
                base_score = base_count * 10  # 基礎分數：每個產品10分
                
                # 獲取該ECCN的特徵配置
                feature_profile = eccn_feature_profiles.get(eccn_code, {})
                
                # 計算特徵匹配度評分
                feature_match_score = 0.0
                matched_features = []
                
                for feature_category, target_data in target_features.items():
                    if feature_category in feature_profile:
                        target_score = target_data.get('normalized_score', 0)
                        expected_score = feature_profile[feature_category]
                        
                        # 計算匹配度 (1 - 差距的絕對值)
                        match_quality = 1.0 - abs(target_score - expected_score)
                        feature_match_score += match_quality * 10  # 每個特徵最多10分
                        
                        if match_quality > 0.7:  # 高匹配度
                            matched_features.append(f"{feature_category}({match_quality:.2f})")
                
                # 規格接近度額外獎勵
                specification_bonus = 0
                
                # 根據特徵匹配度給予規格獎勵（通用邏輯，無特定型號處理）
                # 計算整體特徵複雜度分數
                total_feature_score = sum(
                    target_features.get(category, {}).get('normalized_score', 0) 
                    for category in ['management', 'security', 'protocols', 'quality', 'performance', 'industrial']
                ) / 6.0  # 平均分數
                
                # 根據ECCN級別和特徵複雜度計算規格獎勵
                eccn_complexity_weights = {
                    'EAR99': 0.1,
                    '5A991': 0.4,
                    '5A991.b': 0.7,
                    '5A991.b.1': 0.8,
                    '4A994': 0.6,
                    '5A992.c': 0.9
                }
                
                expected_complexity = eccn_complexity_weights.get(eccn_code, 0.5)
                complexity_match = 1.0 - abs(total_feature_score - expected_complexity)
                
                if complexity_match > 0.7:  # 高匹配度
                    specification_bonus = int(complexity_match * 50)  # 最高50分獎勵
                    self.logger.info(f"📊 {eccn_code}特徵複雜度匹配({complexity_match:.2f})，獲得規格獎勵: +{specification_bonus}")
                elif complexity_match > 0.5:  # 中等匹配度
                    specification_bonus = int(complexity_match * 25)  # 最高25分獎勵
                
                # 計算最終評分
                final_score = base_score + feature_match_score + specification_bonus
                
                eccn_scores[eccn_code] = {
                    'count': base_count,
                    'base_score': base_score,
                    'feature_match_score': feature_match_score,
                    'specification_bonus': specification_bonus,
                    'final_score': final_score,
                    'matched_features': matched_features,
                    'feature_profile_match': {
                        category: {
                            'target': target_features.get(category, {}).get('normalized_score', 0),
                            'expected': feature_profile.get(category, 0),
                            'match_quality': 1.0 - abs(target_features.get(category, {}).get('normalized_score', 0) - feature_profile.get(category, 0))
                        }
                        for category in feature_profile.keys()
                    }
                }
                
                self.logger.info(f"📊 {eccn_code}: 基礎({base_score}) + 特徵匹配({feature_match_score:.1f}) + 規格獎勵({specification_bonus}) = {final_score:.1f}")
            
            return eccn_scores
            
        except Exception as e:
            self.logger.error(f" ECCN評分計算失敗: {str(e)}")
            return {}

# 使用示例和測試
def test_mouser_api():
    """測試Mouser API功能"""
    
    # 設置日誌
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 創建客戶端 (無API金鑰時會使用公開端點)
    client = MouserAPIClient(api_key=None, logger=logger)
    
    # 測試產品
    test_models = [
        "EKI-2428G-4CA-AE",
        "EKI-5525I-AE", 
        "TN-5510A-2L",
        "TN-4500A-T"
    ]
    
    print(" Mouser API 測試")
    print("=" * 50)
    
    for model in test_models:
        print(f"\n 測試型號: {model}")
        
        eccn_info = client.get_eccn_info(model)
        
        if eccn_info:
            print(f" ECCN: {eccn_info.get('eccn_code', 'N/A')}")
            print(f"   信心度: {eccn_info.get('confidence', 'unknown')}")
            print(f"   搜索方法: {eccn_info.get('search_method', 'unknown')}")
            print(f"   製造商: {eccn_info.get('manufacturer', 'N/A')}")
        else:
            print(" 未找到ECCN資訊")
        
        time.sleep(2)  # 避免API速率限制

if __name__ == "__main__":
    test_mouser_api()#!/usr/bin/env python3
"""
WebSearch 整合模組
提供ECCN相關的網路搜索功能
包含官方法規、商業資料庫、技術文檔搜索
"""

import json
import requests
import re
import time
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from urllib.parse import quote, urljoin
import html

class ECCNWebSearcher:
    """ECCN網路搜索器"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # 搜索配置
        self.timeout = 30
        self.max_results_per_query = 10
        self.rate_limit_delay = 1
        
        # ECCN 相關關鍵字和模式
        self.eccn_pattern = re.compile(r'\b(EAR99|[0-9][A-Z][0-9]{3}(?:\.[a-z](?:\.[0-9]+)?)?)\b', re.IGNORECASE)
        
        # 官方和權威來源網域
        self.authoritative_domains = {
            'bis.doc.gov': {'weight': 10, 'type': 'government'},
            'export.gov': {'weight': 10, 'type': 'government'},
            'trade.gov': {'weight': 9, 'type': 'government'},
            'census.gov': {'weight': 8, 'type': 'government'},
            'cbp.gov': {'weight': 8, 'type': 'government'},
            'mouser.com': {'weight': 7, 'type': 'distributor'},
            'digikey.com': {'weight': 7, 'type': 'distributor'},
            'advantech.com': {'weight': 6, 'type': 'manufacturer'},
            'cisco.com': {'weight': 6, 'type': 'manufacturer'},
            'dell.com': {'weight': 6, 'type': 'manufacturer'},
            'hp.com': {'weight': 6, 'type': 'manufacturer'},
            'intel.com': {'weight': 6, 'type': 'manufacturer'}
        }
        
        # 搜索查詢模板
        self.query_templates = {
            'official_eccn': '"{product_model}" ECCN site:bis.doc.gov OR site:export.gov',
            'distributor_eccn': '"{product_model}" ECCN site:mouser.com OR site:digikey.com',
            'manufacturer_eccn': '"{product_model}" "export control" OR "ECCN" site:{manufacturer_domain}',
            'general_eccn': '"{product_model}" ECCN "export control classification"',
            'ccl_search': '"{product_model}" "commerce control list" CCL',
            'regulatory_search': '"{product_model}" "export administration regulations" EAR',
            'specification_search': '"{product_model}" specifications datasheet ECCN'
        }

    def search_eccn_information(self, product_model: str, manufacturer: str = None) -> List[Dict]:
        """
        搜索產品的ECCN相關資訊
        
        Args:
            product_model: 產品型號
            manufacturer: 製造商名稱 (可選)
            
        Returns:
            搜索結果清單
        """
        try:
            self.logger.info(f" 開始WebSearch ECCN查詢: {product_model}")
            
            all_results = []
            
            # 1. 官方政府來源搜索
            official_results = self._search_official_sources(product_model)
            all_results.extend(official_results)
            
            # 2. 經銷商網站搜索
            distributor_results = self._search_distributor_sources(product_model)
            all_results.extend(distributor_results)
            
            # 3. 製造商網站搜索
            if manufacturer:
                manufacturer_results = self._search_manufacturer_sources(product_model, manufacturer)
                all_results.extend(manufacturer_results)
            
            # 4. 一般ECCN資料庫搜索
            general_results = self._search_general_sources(product_model)
            all_results.extend(general_results)
            
            # 5. 分析和排序結果
            processed_results = self._process_search_results(all_results, product_model)
            
            self.logger.info(f" WebSearch完成，找到 {len(processed_results)} 個相關結果")
            return processed_results
            
        except Exception as e:
            self.logger.error(f" WebSearch失敗: {str(e)}")
            return []

    def _search_official_sources(self, product_model: str) -> List[Dict]:
        """搜索官方政府來源"""
        
        self.logger.info("️ 搜索官方政府來源")
        results = []
        
        queries = [
            self.query_templates['official_eccn'].format(product_model=product_model),
            f'"{product_model}" "commerce control list"',
            f'"{product_model}" "export administration regulations"'
        ]
        
        for query in queries:
            try:
                search_results = self._perform_web_search(query, source_type='official')
                results.extend(search_results)
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                self.logger.warning(f"官方來源搜索失敗: {query} - {str(e)}")
        
        return results

    def _search_distributor_sources(self, product_model: str) -> List[Dict]:
        """搜索經銷商來源"""
        
        self.logger.info(" 搜索經銷商來源")
        results = []
        
        queries = [
            self.query_templates['distributor_eccn'].format(product_model=product_model),
            f'"{product_model}" ECCN site:arrow.com',
            f'"{product_model}" "export control" site:farnell.com'
        ]
        
        for query in queries:
            try:
                search_results = self._perform_web_search(query, source_type='distributor')
                results.extend(search_results)
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                self.logger.warning(f"經銷商來源搜索失敗: {query} - {str(e)}")
        
        return results

    def _search_manufacturer_sources(self, product_model: str, manufacturer: str) -> List[Dict]:
        """搜索製造商來源"""
        
        self.logger.info(f" 搜索製造商來源: {manufacturer}")
        results = []
        
        # 推斷製造商網域
        manufacturer_domain = self._infer_manufacturer_domain(manufacturer)
        
        if manufacturer_domain:
            query = self.query_templates['manufacturer_eccn'].format(
                product_model=product_model,
                manufacturer_domain=manufacturer_domain
            )
            
            try:
                search_results = self._perform_web_search(query, source_type='manufacturer')
                results.extend(search_results)
            except Exception as e:
                self.logger.warning(f"製造商來源搜索失敗: {manufacturer} - {str(e)}")
        
        return results

    def _search_general_sources(self, product_model: str) -> List[Dict]:
        """搜索一般來源"""
        
        self.logger.info(" 搜索一般來源")
        results = []
        
        queries = [
            self.query_templates['general_eccn'].format(product_model=product_model),
            self.query_templates['ccl_search'].format(product_model=product_model),
            self.query_templates['specification_search'].format(product_model=product_model)
        ]
        
        for query in queries:
            try:
                search_results = self._perform_web_search(query, source_type='general')
                results.extend(search_results)
                time.sleep(self.rate_limit_delay)
            except Exception as e:
                self.logger.warning(f"一般來源搜索失敗: {query} - {str(e)}")
        
        return results

    def _perform_web_search(self, query: str, source_type: str = 'general') -> List[Dict]:
        """
        執行實際的網路搜索
        
        注意: 這裡使用模擬實現，實際部署時需要整合真實的搜索API
        如 Google Custom Search API, Bing Search API 或 DuckDuckGo API
        """
        
        # 模擬搜索結果 - 實際實現時需要替換為真實的搜索API
        mock_results = self._generate_mock_search_results(query, source_type)
        
        # 真實實現範例 (需要API金鑰):
        # return self._google_custom_search(query)
        # return self._bing_search(query)
        # return self._duckduckgo_search(query)
        
        return mock_results

    def _generate_mock_search_results(self, query: str, source_type: str) -> List[Dict]:
        """生成模擬搜索結果 (開發和測試用)"""
        
        mock_data = {
            'official': [
                {
                    'title': 'Commerce Control List - Category 5 Telecommunications',
                    'url': 'https://www.bis.doc.gov/index.php/policy-guidance/control-list-classification/commerce-control-list-classification',
                    'snippet': 'Industrial networking equipment classified under 5A991. Products with enhanced security features may require 5A991.b classification.',
                    'domain': 'bis.doc.gov'
                }
            ],
            'distributor': [
                {
                    'title': f'Search results for {query.split()[0]} - ECCN Information',
                    'url': 'https://www.mouser.com/ProductDetail/Advantech/EKI-2428G-4CA-AE',
                    'snippet': 'ECCN: 5A991 - Industrial Ethernet networking equipment. Export restrictions may apply.',
                    'domain': 'mouser.com'
                }
            ],
            'manufacturer': [
                {
                    'title': 'Product Specifications and Export Classification',
                    'url': 'https://www.advantech.com/products/eccn-classification',
                    'snippet': 'Our industrial networking products are classified according to US export control regulations. Most industrial switches fall under 5A991.',
                    'domain': 'advantech.com'
                }
            ],
            'general': [
                {
                    'title': 'Understanding ECCN Classifications for Network Equipment',
                    'url': 'https://www.exportcompliance.com/eccn-guide',
                    'snippet': 'Network switches and industrial communication equipment typically classified as 5A991 or EAR99 depending on capabilities.',
                    'domain': 'exportcompliance.com'
                }
            ]
        }
        
        return mock_data.get(source_type, [])

    def _google_custom_search(self, query: str) -> List[Dict]:
        """Google Custom Search API 整合 (需要API金鑰)"""
        
        # Google Custom Search API 實現
        api_key = ""  # 需要設定
        cx = ""  # Custom Search Engine ID
        
        if not api_key or not cx:
            return []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': api_key,
            'cx': cx,
            'q': query,
            'num': self.max_results_per_query
        }
        
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                return [
                    {
                        'title': item.get('title', ''),
                        'url': item.get('link', ''),
                        'snippet': item.get('snippet', ''),
                        'domain': self._extract_domain(item.get('link', ''))
                    }
                    for item in items
                ]
        except Exception as e:
            self.logger.error(f"Google搜索失敗: {str(e)}")
        
        return []

    def _bing_search(self, query: str) -> List[Dict]:
        """Bing Search API 整合 (需要API金鑰)"""
        
        api_key = ""  # 需要設定
        
        if not api_key:
            return []
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {'Ocp-Apim-Subscription-Key': api_key}
        params = {
            'q': query,
            'count': self.max_results_per_query,
            'mkt': 'en-US'
        }
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                web_pages = data.get('webPages', {}).get('value', [])
                
                return [
                    {
                        'title': page.get('name', ''),
                        'url': page.get('url', ''),
                        'snippet': page.get('snippet', ''),
                        'domain': self._extract_domain(page.get('url', ''))
                    }
                    for page in web_pages
                ]
        except Exception as e:
            self.logger.error(f"Bing搜索失敗: {str(e)}")
        
        return []

    def _process_search_results(self, raw_results: List[Dict], product_model: str) -> List[Dict]:
        """處理和分析搜索結果"""
        
        processed_results = []
        
        for result in raw_results:
            try:
                # 提取ECCN資訊
                eccn_info = self._extract_eccn_from_result(result, product_model)
                
                if eccn_info:
                    # 計算相關性和可信度
                    relevance_score = self._calculate_relevance(result, product_model)
                    credibility_score = self._calculate_credibility(result)
                    
                    processed_result = {
                        **eccn_info,
                        'relevance_score': relevance_score,
                        'credibility_score': credibility_score,
                        'combined_score': (relevance_score + credibility_score) / 2,
                        'source_type': self._classify_source_type(result),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    processed_results.append(processed_result)
                    
            except Exception as e:
                self.logger.warning(f"結果處理失敗: {str(e)}")
                continue
        
        # 按分數排序
        processed_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # 去重和合併相似結果
        deduplicated_results = self._deduplicate_results(processed_results)
        
        return deduplicated_results

    def _extract_eccn_from_result(self, result: Dict, product_model: str) -> Optional[Dict]:
        """從搜索結果中提取ECCN資訊"""
        
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        url = result.get('url', '')
        domain = result.get('domain', '')
        
        combined_text = f"{title} {snippet}".upper()
        
        # 尋找ECCN模式
        eccn_matches = self.eccn_pattern.findall(combined_text)
        
        if eccn_matches:
            # 選擇最可能的ECCN (排除明顯不相關的)
            eccn_code = self._select_best_eccn_match(eccn_matches, combined_text)
            
            return {
                'source': 'websearch',
                'eccn_code': eccn_code,
                'url': url,
                'domain': domain,
                'title': title,
                'snippet': snippet,
                'context': self._extract_context(combined_text, eccn_code),
                'confidence': self._assess_eccn_confidence(result, eccn_code, product_model)
            }
        
        return None

    def _select_best_eccn_match(self, eccn_matches: List[str], text: str) -> str:
        """從多個ECCN匹配中選擇最佳的"""
        
        if len(eccn_matches) == 1:
            return eccn_matches[0].upper()
        
        # 優先級排序
        eccn_priorities = {
            'EAR99': 1,
            '5A991': 2,
            '5A991.b': 3,
            '5A991.b.1': 4,
            '4A994': 2
        }
        
        # 按優先級和上下文相關性排序
        scored_matches = []
        for eccn in eccn_matches:
            eccn_upper = eccn.upper()
            priority = eccn_priorities.get(eccn_upper, 0)
            context_score = self._calculate_context_relevance(text, eccn_upper)
            
            scored_matches.append((eccn_upper, priority + context_score))
        
        # 返回最高分的ECCN
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        return scored_matches[0][0]

    def _calculate_context_relevance(self, text: str, eccn: str) -> float:
        """計算ECCN在文本中的上下文相關性"""
        
        # 相關關鍵字
        relevance_keywords = {
            'EAR99': ['commercial', 'consumer', 'office', 'standard'],
            '5A991': ['industrial', 'managed', 'ethernet', 'network'],
            '5A991.b': ['security', 'encryption', 'advanced', 'enhanced'],
            '5A991.b.1': ['high-speed', 'gigabit', 'performance', 'fiber'],
            '4A994': ['management', 'monitoring', 'control', 'power']
        }
        
        keywords = relevance_keywords.get(eccn, [])
        found_keywords = sum(1 for keyword in keywords if keyword in text.lower())
        
        return found_keywords / len(keywords) if keywords else 0

    def _calculate_relevance(self, result: Dict, product_model: str) -> float:
        """計算搜索結果與產品的相關性"""
        
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        url = result.get('url', '').lower()
        
        model_clean = product_model.lower().replace('-', '').replace('_', '')
        
        relevance_score = 0.0
        
        # 產品型號匹配
        if model_clean in title.replace('-', '').replace('_', ''):
            relevance_score += 0.4
        elif model_clean in snippet.replace('-', '').replace('_', ''):
            relevance_score += 0.3
        elif model_clean in url.replace('-', '').replace('_', ''):
            relevance_score += 0.2
        
        # ECCN相關關鍵字
        eccn_keywords = ['eccn', 'export control', 'classification', 'commerce control list']
        for keyword in eccn_keywords:
            if keyword in title or keyword in snippet:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)

    def _calculate_credibility(self, result: Dict) -> float:
        """計算搜索結果的可信度"""
        
        domain = result.get('domain', '').lower()
        
        # 檢查是否為權威來源
        for auth_domain, info in self.authoritative_domains.items():
            if auth_domain in domain:
                return info['weight'] / 10.0  # 標準化為0-1
        
        # 其他可信度指標
        credibility_score = 0.5  # 基礎分數
        
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        
        # 正面指標
        positive_indicators = ['official', 'specification', 'datasheet', 'technical', 'manufacturer']
        negative_indicators = ['forum', 'discussion', 'blog', 'opinion', 'estimate']
        
        for indicator in positive_indicators:
            if indicator in title or indicator in snippet:
                credibility_score += 0.1
        
        for indicator in negative_indicators:
            if indicator in title or indicator in snippet:
                credibility_score -= 0.1
        
        return max(0.0, min(credibility_score, 1.0))

    def _classify_source_type(self, result: Dict) -> str:
        """分類來源類型"""
        
        domain = result.get('domain', '').lower()
        
        for auth_domain, info in self.authoritative_domains.items():
            if auth_domain in domain:
                return info['type']
        
        # 根據域名特徵分類
        if any(gov_tld in domain for gov_tld in ['.gov', '.mil']):
            return 'government'
        elif any(edu_tld in domain for edu_tld in ['.edu', '.ac.']):
            return 'academic'
        elif 'export' in domain or 'compliance' in domain:
            return 'compliance'
        else:
            return 'general'

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """去重相似的搜索結果"""
        
        unique_results = []
        seen_eccns = set()
        
        for result in results:
            eccn = result.get('eccn_code', '')
            url = result.get('url', '')
            
            # 組合唯一鍵
            unique_key = f"{eccn}_{self._extract_domain(url)}"
            
            if unique_key not in seen_eccns:
                seen_eccns.add(unique_key)
                unique_results.append(result)
        
        return unique_results

    def _extract_domain(self, url: str) -> str:
        """從URL提取域名"""
        
        import urllib.parse
        try:
            parsed = urllib.parse.urlparse(url)
            return parsed.netloc.lower()
        except:
            return ''

    def _extract_context(self, text: str, eccn: str) -> str:
        """提取ECCN周圍的上下文"""
        
        eccn_pos = text.upper().find(eccn.upper())
        if eccn_pos == -1:
            return ''
        
        # 提取前後50個字符
        start = max(0, eccn_pos - 50)
        end = min(len(text), eccn_pos + len(eccn) + 50)
        
        return text[start:end].strip()

    def _assess_eccn_confidence(self, result: Dict, eccn: str, product_model: str) -> str:
        """評估ECCN資訊的信心度"""
        
        relevance = self._calculate_relevance(result, product_model)
        credibility = self._calculate_credibility(result)
        
        combined_score = (relevance + credibility) / 2
        
        if combined_score >= 0.8:
            return 'high'
        elif combined_score >= 0.6:
            return 'medium'
        else:
            return 'low'

    def _infer_manufacturer_domain(self, manufacturer: str) -> Optional[str]:
        """推斷製造商的網域名稱"""
        
        manufacturer_mapping = {
            'advantech': 'advantech.com',
            'cisco': 'cisco.com',
            'dell': 'dell.com',
            'hp': 'hp.com',
            'intel': 'intel.com',
            'siemens': 'siemens.com',
            'schneider': 'schneider-electric.com',
            'moxa': 'moxa.com',
            'phoenix': 'phoenixcontact.com'
        }
        
        manufacturer_lower = manufacturer.lower()
        
        for key, domain in manufacturer_mapping.items():
            if key in manufacturer_lower:
                return domain
        
        # 嘗試構建域名
        clean_name = re.sub(r'[^a-zA-Z]', '', manufacturer_lower)
        if len(clean_name) > 2:
            return f"{clean_name}.com"
        
        return None

# 使用示例
def test_websearch():
    """測試WebSearch功能"""
    
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    searcher = ECCNWebSearcher(logger)
    
    test_products = [
        ("EKI-2428G-4CA-AE", "Advantech"),
        ("EKI-5525I-AE", "Advantech"),
        ("TN-5510A-2L", "Moxa")
    ]
    
    print(" WebSearch 測試")
    print("=" * 50)
    
    for model, manufacturer in test_products:
        print(f"\n 搜索: {model} ({manufacturer})")
        
        results = searcher.search_eccn_information(model, manufacturer)
        
        print(f"找到 {len(results)} 個結果:")
        for i, result in enumerate(results[:3], 1):  # 顯示前3個結果
            print(f"  {i}. ECCN: {result.get('eccn_code', 'N/A')}")
            print(f"     信心度: {result.get('confidence', 'unknown')}")
            print(f"     來源: {result.get('domain', 'unknown')}")
            print(f"     分數: {result.get('combined_score', 0):.2f}")

if __name__ == "__main__":
    test_websearch()