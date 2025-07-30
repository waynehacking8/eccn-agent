#!/usr/bin/env python3
"""
Mouser API Similar Product Comparison Algorithm
Focus on technical specification similarity and ECCN distribution analysis
"""

import json
import requests
import re
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import Counter
import difflib

class MouserSimilarityAnalyzer:
    """
    Mouser API similarity analyzer focusing on:
    1. Technical specification matching
    2. Smart ECCN distribution analysis
    3. Product feature comparison
    4. Confidence scoring based on similarity metrics
    """
    
    def __init__(self, api_key: str = "d337cdb2-f839-4405-b34a-2533df7c60af", logger: logging.Logger = None):
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        
        # Search configuration
        self.search_config = {
            'max_results_per_query': 20,
            'max_total_results': 50,
            'similarity_threshold': 0.7,
            'min_products_for_analysis': 3
        }
        
        # Technical specification patterns
        self.spec_patterns = {
            'temperature': re.compile(r'(-?\d+)°C\s*(?:to|~|-)\s*\+?(\d+)°C', re.IGNORECASE),
            'power_dc': re.compile(r'(\d+)V?\s*DC|DC\s*(\d+)V?', re.IGNORECASE),
            'power_ac': re.compile(r'(\d+)-(\d+)V?\s*AC|AC\s*(\d+)-(\d+)V?', re.IGNORECASE),
            'ports': re.compile(r'(\d+)\s*port', re.IGNORECASE),
            'speed': re.compile(r'(\d+)\s*(Mbps|Gbps)', re.IGNORECASE),
            'management': re.compile(r'(SNMP|Web GUI|CLI|managed)', re.IGNORECASE),
            'mounting': re.compile(r'(DIN-rail|rack|desktop)', re.IGNORECASE)
        }
        
        # ECCN classification hierarchy for intelligent scoring
        self.eccn_hierarchy = {
            'EAR99': {'level': 0, 'commercial': True, 'industrial': False},
            '4A994': {'level': 1, 'commercial': False, 'industrial': True},
            '5A991': {'level': 2, 'commercial': False, 'industrial': True},
            '5A991.b': {'level': 3, 'commercial': False, 'industrial': True},
            '5A991.b.1': {'level': 4, 'commercial': False, 'industrial': True},
            '5A992.c': {'level': 5, 'commercial': False, 'industrial': True}
        }
    
    def analyze_similar_products(self, pdf_content: str, product_model: str) -> Dict:
        """
        Main entry point for similarity analysis
        """
        try:
            self.logger.info(f"Starting similarity analysis for {product_model}")
            
            # Step 1: Extract technical specifications from PDF
            target_specs = self._extract_technical_specs(pdf_content)
            self.logger.info(f"Extracted specs: {target_specs}")
            
            # Step 2: Generate smart search queries based on specs
            search_queries = self._generate_smart_queries(target_specs, product_model)
            
            # Step 3: Search for similar products using multiple strategies
            similar_products = self._search_with_multiple_strategies(search_queries)
            
            if len(similar_products) < self.search_config['min_products_for_analysis']:
                self.logger.warning(f"Only found {len(similar_products)} products, insufficient for analysis")
                return self._create_failure_response("Insufficient similar products found")
            
            # Step 4: Calculate similarity scores for each product
            scored_products = self._calculate_similarity_scores(similar_products, target_specs)
            
            # Step 5: Analyze ECCN distribution with confidence weighting
            eccn_analysis = self._analyze_eccn_distribution(scored_products, target_specs)
            
            # Step 6: Generate final recommendation
            return self._generate_final_recommendation(eccn_analysis, scored_products, target_specs)
            
        except Exception as e:
            self.logger.error(f"Similarity analysis failed: {str(e)}")
            return self._create_failure_response(str(e))
    
    def _extract_technical_specs(self, pdf_content: str) -> Dict:
        """Extract technical specifications from PDF content"""
        specs = {
            'temperature_range': None,
            'power_type': None,
            'power_specs': {},
            'port_count': None,
            'speed_specs': [],
            'management_features': [],
            'mounting_type': None,
            'industrial_features': []
        }
        
        # Temperature range extraction
        temp_match = self.spec_patterns['temperature'].search(pdf_content)
        if temp_match:
            min_temp = int(temp_match.group(1))
            max_temp = int(temp_match.group(2))
            specs['temperature_range'] = (min_temp, max_temp)
            
            # Classify temperature range
            if min_temp <= -20 and max_temp >= 70:
                specs['industrial_features'].append('wide_temperature')
        
        # Power specifications
        dc_match = self.spec_patterns['power_dc'].search(pdf_content)
        ac_match = self.spec_patterns['power_ac'].search(pdf_content)
        
        if dc_match:
            specs['power_type'] = 'DC'
            voltage = dc_match.group(1) or dc_match.group(2)
            if voltage:
                specs['power_specs']['dc_voltage'] = int(voltage)
        elif ac_match:
            specs['power_type'] = 'AC'
            specs['power_specs']['ac_range'] = f"{ac_match.group(1)}-{ac_match.group(2)}V"
        
        # Port count
        port_match = self.spec_patterns['ports'].search(pdf_content)
        if port_match:
            specs['port_count'] = int(port_match.group(1))
        
        # Speed specifications
        for speed_match in self.spec_patterns['speed'].finditer(pdf_content):
            speed_value = int(speed_match.group(1))
            speed_unit = speed_match.group(2).lower()
            if speed_unit == 'gbps' or (speed_unit == 'mbps' and speed_value >= 1000):
                specs['speed_specs'].append(f"{speed_value} {speed_unit}")
        
        # Management features
        if self.spec_patterns['management'].search(pdf_content):
            specs['management_features'].append('managed')
            if 'SNMP' in pdf_content.upper():
                specs['management_features'].append('snmp')
            if 'WEB' in pdf_content.upper() and 'GUI' in pdf_content.upper():
                specs['management_features'].append('web_gui')
        
        # Mounting type
        mount_match = self.spec_patterns['mounting'].search(pdf_content)
        if mount_match:
            specs['mounting_type'] = mount_match.group(1).lower()
            if 'din-rail' in specs['mounting_type']:
                specs['industrial_features'].append('din_rail')
        
        return specs
    
    def _generate_smart_queries(self, specs: Dict, product_model: str) -> List[str]:
        """Generate intelligent search queries based on extracted specifications"""
        queries = []
        
        # Base product type queries
        if specs['management_features']:
            queries.append("managed ethernet switch")
            queries.append("industrial managed switch")
        else:
            queries.append("unmanaged ethernet switch")
            queries.append("ethernet switch")
        
        # Temperature-based queries
        if specs['temperature_range']:
            min_temp, max_temp = specs['temperature_range']
            if min_temp <= -20:
                queries.append("industrial temperature ethernet switch")
                queries.append("wide temperature switch")
        
        # Power-based queries
        if specs['power_type'] == 'DC':
            queries.append("DC power ethernet switch")
            queries.append("industrial DC switch")
        
        # Port-based queries
        if specs['port_count']:
            queries.append(f"{specs['port_count']} port ethernet switch")
        
        # Speed-based queries
        if specs['speed_specs']:
            if any('gbps' in speed.lower() for speed in specs['speed_specs']):
                queries.append("gigabit ethernet switch")
        
        # Industrial features queries
        if 'din_rail' in specs['industrial_features']:
            queries.append("DIN rail ethernet switch")
        
        # Manufacturer-specific query
        if 'EKI' in product_model:
            queries.append("Advantech ethernet switch")
        
        # Remove duplicates and limit
        return list(dict.fromkeys(queries))[:6]
    
    def _search_with_multiple_strategies(self, queries: List[str]) -> List[Dict]:
        """Search using multiple strategies and combine results"""
        all_products = []
        
        for query in queries:
            try:
                self.logger.info(f"Searching: {query}")
                results = self._search_mouser_api(query)
                all_products.extend(results)
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Search failed for '{query}': {str(e)}")
                continue
        
        # Remove duplicates based on part number
        seen_parts = set()
        unique_products = []
        
        for product in all_products:
            part_number = product.get('MouserPartNumber', '')
            if part_number and part_number not in seen_parts:
                seen_parts.add(part_number)
                unique_products.append(product)
        
        self.logger.info(f"Found {len(unique_products)} unique products")
        return unique_products[:self.search_config['max_total_results']]
    
    def _search_mouser_api(self, query: str) -> List[Dict]:
        """Execute Mouser API search"""
        url = f"https://api.mouser.com/api/v1.0/search/keyword?apiKey={self.api_key}"
        
        payload = {
            "SearchByKeywordRequest": {
                "keyword": query,
                "records": self.search_config['max_results_per_query'],
                "startingRecord": 0
            }
        }
        
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            search_results = data.get('SearchResults', {})
            return search_results.get('Parts', [])
        else:
            raise Exception(f"Mouser API error: {response.status_code}")
    
    def _calculate_similarity_scores(self, products: List[Dict], target_specs: Dict) -> List[Dict]:
        """Calculate detailed similarity scores for each product"""
        scored_products = []
        
        for product in products:
            try:
                # Extract product specifications
                product_specs = self._extract_product_specs(product)
                
                # Calculate similarity score
                similarity_score = self._compute_similarity_score(product_specs, target_specs)
                
                # Extract ECCN if available
                eccn_code = self._extract_eccn_from_product(product)
                
                if eccn_code and similarity_score >= self.search_config['similarity_threshold']:
                    scored_products.append({
                        'product': product,
                        'specifications': product_specs,
                        'similarity_score': similarity_score,
                        'eccn_code': eccn_code,
                        'confidence': self._calculate_confidence(similarity_score, product_specs)
                    })
                    
            except Exception as e:
                self.logger.warning(f"Failed to score product: {str(e)}")
                continue
        
        # Sort by similarity score
        scored_products.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        self.logger.info(f"Scored {len(scored_products)} products above threshold")
        return scored_products
    
    def _compute_similarity_score(self, product_specs: Dict, target_specs: Dict) -> float:
        """Compute detailed similarity score between products"""
        score_components = {}
        total_weight = 0
        
        # Temperature range similarity (weight: 30%)
        if target_specs['temperature_range'] and product_specs['temperature_range']:
            temp_similarity = self._calculate_temperature_similarity(
                target_specs['temperature_range'], 
                product_specs['temperature_range']
            )
            score_components['temperature'] = temp_similarity * 0.30
            total_weight += 0.30
        
        # Power specifications similarity (weight: 25%)
        if target_specs['power_type'] and product_specs['power_type']:
            power_similarity = 1.0 if target_specs['power_type'] == product_specs['power_type'] else 0.5
            score_components['power'] = power_similarity * 0.25
            total_weight += 0.25
        
        # Port count similarity (weight: 15%)
        if target_specs['port_count'] and product_specs['port_count']:
            port_similarity = self._calculate_port_similarity(
                target_specs['port_count'], 
                product_specs['port_count']
            )
            score_components['ports'] = port_similarity * 0.15
            total_weight += 0.15
        
        # Management features similarity (weight: 20%)
        if target_specs['management_features'] or product_specs['management_features']:
            mgmt_similarity = self._calculate_feature_similarity(
                target_specs['management_features'], 
                product_specs['management_features']
            )
            score_components['management'] = mgmt_similarity * 0.20
            total_weight += 0.20
        
        # Industrial features similarity (weight: 10%)
        if target_specs['industrial_features'] or product_specs['industrial_features']:
            industrial_similarity = self._calculate_feature_similarity(
                target_specs['industrial_features'], 
                product_specs['industrial_features']
            )
            score_components['industrial'] = industrial_similarity * 0.10
            total_weight += 0.10
        
        # Calculate final score
        if total_weight > 0:
            final_score = sum(score_components.values()) / total_weight
        else:
            final_score = 0.0
        
        return min(final_score, 1.0)
    
    def _calculate_temperature_similarity(self, temp1: Tuple[int, int], temp2: Tuple[int, int]) -> float:
        """Calculate temperature range similarity"""
        min1, max1 = temp1
        min2, max2 = temp2
        
        # Calculate overlap percentage
        overlap_min = max(min1, min2)
        overlap_max = min(max1, max2)
        
        if overlap_max <= overlap_min:
            return 0.0  # No overlap
        
        overlap_range = overlap_max - overlap_min
        target_range = max1 - min1
        product_range = max2 - min2
        
        # Similarity based on overlap and range difference
        overlap_score = overlap_range / max(target_range, product_range)
        range_difference = abs(target_range - product_range) / max(target_range, product_range)
        
        return overlap_score * (1 - range_difference * 0.5)
    
    def _calculate_port_similarity(self, ports1: int, ports2: int) -> float:
        """Calculate port count similarity"""
        if ports1 == ports2:
            return 1.0
        
        diff = abs(ports1 - ports2)
        max_ports = max(ports1, ports2)
        
        # Penalize larger differences more
        return max(0.0, 1.0 - (diff / max_ports))
    
    def _calculate_feature_similarity(self, features1: List[str], features2: List[str]) -> float:
        """Calculate feature set similarity using Jaccard index"""
        if not features1 and not features2:
            return 1.0
        
        set1 = set(features1)
        set2 = set(features2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_eccn_distribution(self, scored_products: List[Dict], target_specs: Dict) -> Dict:
        """Analyze ECCN distribution with confidence weighting"""
        if not scored_products:
            return {}
        
        # Weight ECCN votes by similarity score and confidence
        eccn_weighted_votes = {}
        total_weight = 0
        
        for product_data in scored_products:
            eccn = product_data['eccn_code']
            similarity = product_data['similarity_score']
            confidence = product_data['confidence']
            
            # Calculate vote weight (similarity * confidence)
            weight = similarity * confidence
            
            if eccn not in eccn_weighted_votes:
                eccn_weighted_votes[eccn] = {
                    'weight': 0,
                    'count': 0,
                    'products': []
                }
            
            eccn_weighted_votes[eccn]['weight'] += weight
            eccn_weighted_votes[eccn]['count'] += 1
            eccn_weighted_votes[eccn]['products'].append(product_data)
            total_weight += weight
        
        # Calculate normalized scores
        eccn_analysis = {}
        for eccn, data in eccn_weighted_votes.items():
            normalized_score = data['weight'] / total_weight if total_weight > 0 else 0
            
            eccn_analysis[eccn] = {
                'count': data['count'],
                'weight': data['weight'],
                'normalized_score': normalized_score,
                'percentage': (data['count'] / len(scored_products)) * 100,
                'avg_similarity': sum(p['similarity_score'] for p in data['products']) / len(data['products']),
                'products': data['products']
            }
        
        return eccn_analysis
    
    def _generate_final_recommendation(self, eccn_analysis: Dict, scored_products: List[Dict], target_specs: Dict) -> Dict:
        """Generate final ECCN recommendation with detailed reasoning"""
        if not eccn_analysis:
            return self._create_failure_response("No ECCN analysis available")
        
        # Find best ECCN based on normalized score
        best_eccn = max(eccn_analysis.keys(), key=lambda x: eccn_analysis[x]['normalized_score'])
        best_analysis = eccn_analysis[best_eccn]
        
        # Determine confidence level
        if best_analysis['normalized_score'] >= 0.7 and best_analysis['count'] >= 3:
            confidence = 'high'
        elif best_analysis['normalized_score'] >= 0.5 and best_analysis['count'] >= 2:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        # Generate detailed reasoning
        reasoning_parts = [
            f"Based on analysis of {len(scored_products)} similar products",
            f"{best_analysis['count']} products ({best_analysis['percentage']:.1f}%) classified as {best_eccn}",
            f"Average similarity score: {best_analysis['avg_similarity']:.2f}",
            f"Weighted confidence: {best_analysis['normalized_score']:.2f}"
        ]
        
        reasoning = ". ".join(reasoning_parts)
        
        return {
            'status': 'success',
            'eccn_code': best_eccn,
            'confidence': confidence,
            'reasoning': reasoning,
            'method': 'similarity_analysis',
            'similar_products_count': len(scored_products),
            'eccn_distribution': {
                eccn: {
                    'count': data['count'],
                    'percentage': data['percentage'],
                    'avg_similarity': data['avg_similarity']
                }
                for eccn, data in eccn_analysis.items()
            },
            'top_similar_products': [
                {
                    'part_number': p['product'].get('MouserPartNumber', ''),
                    'description': p['product'].get('Description', '')[:100],
                    'eccn': p['eccn_code'],
                    'similarity_score': p['similarity_score']
                }
                for p in scored_products[:5]
            ],
            'technical_analysis': {
                'target_specifications': target_specs,
                'matching_criteria': self._summarize_matching_criteria(target_specs)
            }
        }
    
    def _extract_product_specs(self, product: Dict) -> Dict:
        """Extract specifications from Mouser product data"""
        description = product.get('Description', '')
        datasheet_url = product.get('DataSheetUrl', '')
        
        # This would ideally fetch and parse the datasheet
        # For now, extract from description
        specs = {
            'temperature_range': None,
            'power_type': None,
            'port_count': None,
            'management_features': [],
            'industrial_features': []
        }
        
        # Basic pattern matching on description
        temp_match = self.spec_patterns['temperature'].search(description)
        if temp_match:
            specs['temperature_range'] = (int(temp_match.group(1)), int(temp_match.group(2)))
        
        if self.spec_patterns['power_dc'].search(description):
            specs['power_type'] = 'DC'
        elif self.spec_patterns['power_ac'].search(description):
            specs['power_type'] = 'AC'
        
        port_match = self.spec_patterns['ports'].search(description)
        if port_match:
            specs['port_count'] = int(port_match.group(1))
        
        if self.spec_patterns['management'].search(description):
            specs['management_features'].append('managed')
        
        if 'din-rail' in description.lower():
            specs['industrial_features'].append('din_rail')
        
        return specs
    
    def _extract_eccn_from_product(self, product: Dict) -> Optional[str]:
        """Extract ECCN from product data"""
        # Check various fields where ECCN might be stored
        eccn_fields = ['ECCN', 'ExportControlClassificationNumber', 'ECCNCode']
        
        for field in eccn_fields:
            if field in product and product[field]:
                eccn = str(product[field]).strip()
                if self._is_valid_eccn(eccn):
                    return eccn
        
        # Check in description or other text fields
        description = product.get('Description', '')
        eccn_match = re.search(r'\b(EAR99|[0-9][A-Z][0-9]{3}(?:\.[a-zA-Z](?:\.[0-9]+)?)?)\b', description)
        if eccn_match:
            return eccn_match.group(1)
        
        return None
    
    def _is_valid_eccn(self, eccn: str) -> bool:
        """Validate ECCN format"""
        if not eccn:
            return False
        
        eccn = eccn.strip().upper()
        return eccn == 'EAR99' or re.match(r'^[0-9][A-Z][0-9]{3}(?:\.[a-zA-Z](?:\.[0-9]+)?)?$', eccn)
    
    def _calculate_confidence(self, similarity_score: float, product_specs: Dict) -> float:
        """Calculate confidence score for a product match"""
        # Base confidence from similarity
        confidence = similarity_score
        
        # Boost confidence if key specs are available
        if product_specs.get('temperature_range'):
            confidence += 0.1
        if product_specs.get('power_type'):
            confidence += 0.1
        if product_specs.get('management_features'):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _summarize_matching_criteria(self, specs: Dict) -> List[str]:
        """Summarize the matching criteria used"""
        criteria = []
        
        if specs['temperature_range']:
            min_temp, max_temp = specs['temperature_range']
            criteria.append(f"Temperature range: {min_temp}°C to {max_temp}°C")
        
        if specs['power_type']:
            criteria.append(f"Power type: {specs['power_type']}")
        
        if specs['port_count']:
            criteria.append(f"Port count: {specs['port_count']}")
        
        if specs['management_features']:
            criteria.append(f"Management: {', '.join(specs['management_features'])}")
        
        if specs['industrial_features']:
            criteria.append(f"Industrial features: {', '.join(specs['industrial_features'])}")
        
        return criteria
    
    def _create_failure_response(self, error_message: str) -> Dict:
        """Create standardized failure response"""
        return {
            'status': 'failed',
            'error': error_message,
            'eccn_code': None,
            'confidence': 'none',
            'method': 'similarity_analysis'
        }

# Usage example and integration point
def integrate_with_existing_system():
    """
    Integration function to replace the existing Mouser similarity search
    """
    analyzer = MouserSimilarityAnalyzer()
    
    # This would replace the existing _mouser_similar_search method
    def mouser_similar_search(pdf_content: str, product_model: str) -> Dict:
        result = analyzer.analyze_similar_products(pdf_content, product_model)
        
        if result['status'] == 'success':
            return {
                'status': 'success',
                'eccn_suggestions': [result],
                'similar_products_count': result['similar_products_count'],
                'method': 'mouser_similarity',
                'confidence': result['confidence']
            }
        else:
            return {
                'status': 'failed',
                'error': result.get('error', 'Unknown error'),
                'eccn_suggestions': [],
                'method': 'mouser_similarity'
            }
    
    return mouser_similar_search