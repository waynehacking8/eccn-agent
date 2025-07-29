#!/usr/bin/env python3
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
            self.logger.info(f"🔍 開始WebSearch ECCN查詢: {product_model}")
            
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
            
            # 計算包含PENDING_VERIFICATION的結果數量
            pending_count = sum(1 for r in processed_results if r.get('eccn_code') == 'PENDING_VERIFICATION')
            confirmed_count = len(processed_results) - pending_count
            
            self.logger.info(f"✅ WebSearch完成，找到 {confirmed_count} 個確認結果，{pending_count} 個待驗證結果")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"❌ WebSearch失敗: {str(e)}")
            return []

    def _search_official_sources(self, product_model: str) -> List[Dict]:
        """搜索官方政府來源"""
        
        self.logger.info("🏛️ 搜索官方政府來源")
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
        
        self.logger.info("🏪 搜索經銷商來源")
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
        
        self.logger.info(f"🏭 搜索製造商來源: {manufacturer}")
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
        
        self.logger.info("🌐 搜索一般來源")
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
        使用DuckDuckGo進行真實搜索，避免模擬結果干擾
        """
        
        self.logger.info(f"🔍 執行真實WebSearch查詢: {query}")
        
        try:
            # 使用 DuckDuckGo HTML 搜索 (無需API金鑰)
            return self._duckduckgo_search(query)
        except Exception as e:
            self.logger.error(f"❌ WebSearch查詢失敗: {str(e)}")
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
        
        # 如果沒有找到ECCN模式，但是來自權威經銷商，記錄為待查證
        if self._is_authoritative_distributor(domain) and self._contains_product_model(combined_text, product_model):
            return {
                'source': 'websearch',
                'eccn_code': 'PENDING_VERIFICATION',  # 需要進一步查證
                'url': url,
                'domain': domain,
                'title': title,
                'snippet': snippet,
                'context': f"Found on authoritative distributor {domain} - requires page content analysis",
                'confidence': 'medium',
                'requires_page_scraping': True
            }
        
        return None

    def _select_best_eccn_match(self, eccn_matches: List[str], text: str) -> str:
        """從多個ECCN匹配中選擇最佳的"""
        
        if len(eccn_matches) == 1:
            return eccn_matches[0].upper()
        
        # 優先級排序 (提高 5A992.c 的優先級)
        eccn_priorities = {
            'EAR99': 1,
            '5A991': 2,
            '5A991.b': 3,
            '5A991.b.1': 4,
            '4A994': 2,
            '5A992.c': 5,  # 最高優先級
            '5A992.C': 5   # 大寫版本
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
        
        # 相關關鍵字 (加強 5A992.c 的識別)
        relevance_keywords = {
            'EAR99': ['commercial', 'consumer', 'office', 'standard'],
            '5A991': ['industrial', 'managed', 'ethernet', 'network'],
            '5A991.b': ['security', 'encryption', 'advanced', 'enhanced'],
            '5A991.b.1': ['high-speed', 'gigabit', 'performance', 'fiber'],
            '4A994': ['management', 'monitoring', 'control', 'power'],
            '5A992.c': ['comprehensive', 'high-end', 'advanced', 'managed', 'security', 'performance', 'protocols'],
            '5A992.C': ['comprehensive', 'high-end', 'advanced', 'managed', 'security', 'performance', 'protocols']
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

    def _is_authoritative_distributor(self, domain: str) -> bool:
        """檢查是否為權威經銷商域名"""
        authoritative_distributors = [
            'mouser.com', 'digikey.com', 'arrow.com', 'farnell.com',
            'element14.com', 'rs-online.com', 'newark.com'
        ]
        
        return any(dist in domain.lower() for dist in authoritative_distributors)

    def _contains_product_model(self, text: str, product_model: str) -> bool:
        """檢查文本是否包含產品型號"""
        text_clean = text.replace('-', '').replace('_', '').upper()
        model_clean = product_model.replace('-', '').replace('_', '').upper()
        
        return model_clean in text_clean

    def _duckduckgo_search(self, query: str) -> List[Dict]:
        """使用DuckDuckGo進行真實網路搜索"""
        
        try:
            import urllib.parse
            from bs4 import BeautifulSoup
            
            # DuckDuckGo搜索URL
            encoded_query = urllib.parse.quote_plus(query)
            search_url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            self.logger.info(f"🔍 DuckDuckGo搜索: {query}")
            
            response = requests.get(search_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # 解析搜索結果 - 修復DuckDuckGo HTML結構解析
            for result_div in soup.find_all('div', class_='result'):
                try:
                    # 找第一個有效連結作為主要結果
                    title_element = None
                    for link in result_div.find_all('a'):
                        if link.get('href') and link.get_text(strip=True):
                            title_element = link
                            break
                    
                    if not title_element:
                        continue
                        
                    title = title_element.get_text(strip=True)
                    url = title_element.get('href', '')
                    
                    # 處理DuckDuckGo重定向URL
                    if url.startswith('//duckduckgo.com/l/?uddg='):
                        # 解碼重定向URL
                        url = urllib.parse.unquote(url.split('uddg=')[1])
                    elif url.startswith('//'):
                        url = 'https:' + url
                    
                    # 提取摘要 - 從result__body中查找文本
                    snippet = ''
                    body_element = result_div.find('div', class_='result__body')
                    if body_element:
                        # 取得所有文本但排除連結文字
                        all_text = body_element.get_text(separator=' ', strip=True)
                        snippet = all_text.replace(title, '').strip()[:200]
                    
                    # 提取域名
                    domain = self._extract_domain(url)
                    
                    if title and url and domain:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'domain': domain
                        })
                        
                        if len(results) >= self.max_results_per_query:
                            break
                            
                except Exception as e:
                    self.logger.warning(f"解析搜索結果失敗: {str(e)}")
                    continue
            
            self.logger.info(f"✅ DuckDuckGo搜索完成，找到 {len(results)} 個結果")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ DuckDuckGo搜索失敗: {str(e)}")
            return []

    def _google_custom_search(self, query: str) -> List[Dict]:
        """Google Custom Search API實現 (需要API金鑰)"""
        
        # 這裡可以實現Google Custom Search API
        # 需要GOOGLE_API_KEY和GOOGLE_CSE_ID
        self.logger.warning("Google Custom Search未實現，需要API金鑰")
        return []

    def _bing_search(self, query: str) -> List[Dict]:
        """Bing Search API實現 (需要API金鑰)"""
        
        # 這裡可以實現Bing Search API
        # 需要BING_SUBSCRIPTION_KEY
        self.logger.warning("Bing Search未實現，需要API金鑰")
        return []

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
    
    print("🔍 WebSearch 測試")
    print("=" * 50)
    
    for model, manufacturer in test_products:
        print(f"\n🔍 搜索: {model} ({manufacturer})")
        
        results = searcher.search_eccn_information(model, manufacturer)
        
        print(f"找到 {len(results)} 個結果:")
        for i, result in enumerate(results[:3], 1):  # 顯示前3個結果
            print(f"  {i}. ECCN: {result.get('eccn_code', 'N/A')}")
            print(f"     信心度: {result.get('confidence', 'unknown')}")
            print(f"     來源: {result.get('domain', 'unknown')}")
            print(f"     分數: {result.get('combined_score', 0):.2f}")

if __name__ == "__main__":
    test_websearch()