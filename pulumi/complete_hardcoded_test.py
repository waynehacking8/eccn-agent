#!/usr/bin/env python3
"""
完整硬編碼測試 - 參考quick_test_fixed.py模式
直接硬編碼所有實際存在的PDF文件路徑
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

# 配置
GROUND_TRUTH_FILE = "../src/sagemaker/Product_proposal_55.xlsx"
PDF_PARSER_URL = "https://uk77kivopn5ivjsjjyci4uewha0erpwb.lambda-url.us-east-1.on.aws/"

def load_ground_truth():
    """載入ground truth數據"""
    try:
        df = pd.read_excel(GROUND_TRUTH_FILE)
        ground_truth = {}
        for _, row in df.iterrows():
            product_model = str(row.get('Manufacturer Part Number (Primary MPN)', '')).strip()
            eccn = str(row.get('US Export Control Classification Number (ECCN) ', '')).strip()
            if product_model and eccn and eccn != 'nan':
                ground_truth[product_model] = eccn
        return ground_truth
    except Exception as e:
        print(f"載入 ground truth 錯誤: {e}")
        return {}

def test_single_product(product_model, pdf_path, expected_eccn):
    """測試單個產品"""
    print(f"\n 測試: {product_model}")
    print(f"  PDF: {Path(pdf_path).name}")
    print(f"  期望: {expected_eccn}")
    
    start_time = time.time()
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            data = {'product_model': product_model}
            
            response = requests.post(
                PDF_PARSER_URL,
                files=files,
                data=data,
                timeout=10800  # 3小時單個請求超時
            )
            
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                if response_data.get('success'):
                    predicted_eccn = response_data.get('eccn_classification', {}).get('eccn_code')
                    method = response_data.get('eccn_classification', {}).get('method')
                    confidence = response_data.get('eccn_classification', {}).get('confidence')
                    
                    # 檢查是否匹配
                    if predicted_eccn == expected_eccn:
                        print(f"   正確: {predicted_eccn} ({method}, {confidence}) - {processing_time:.2f}s")
                        return True, predicted_eccn, method, processing_time
                    else:
                        print(f"   錯誤: 預測 {predicted_eccn}, 期望 {expected_eccn} ({method}, {confidence}) - {processing_time:.2f}s")
                        return False, predicted_eccn, method, processing_time
                else:
                    print(f"   失敗: {response_data.get('error', 'Unknown error')}")
                    return False, None, None, processing_time
            except json.JSONDecodeError as e:
                print(f"   JSON錯誤: {e}")
                return False, None, None, processing_time
        else:
            print(f"  HTTP錯誤: {response.status_code}")
            return False, None, None, processing_time
            
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"  異常: {e}")
        return False, None, None, processing_time

def main():
    """主程序"""
    print("完整硬編碼測試 - 修復後的ECCN分類系統")
    print("=" * 60)
    
    # 載入ground truth
    ground_truth = load_ground_truth()
    if not ground_truth:
        print("無法載入ground truth數據")
        return
    
    print(f"載入 {len(ground_truth)} 個ground truth記錄")
    
    # 硬編碼所有實際存在的PDF測試案例 (52個，包含新增的7個PDF)
    test_cases = [
        ("EKI-2525LI-AE", "../src/sagemaker/data/EKI-2525LI-AE/EKI-2525LI_DS(20191015)20191016105154.pdf"),
        ("EKI-5528I-AE", "../src/sagemaker/data/EKI-5528I-AE/EKI-5528I20150714145225.pdf"),
        ("EKI-5729FI-MB-AE", "../src/sagemaker/data/EKI-5729FI-MB-AE/EKI-5729FI-MB_DS(081921)20210917164354.pdf"),
        ("EKI-2525MI-ST-BE", "../src/sagemaker/data/EKI-2525MI-ST-BE/EKI-2525MI_SI_DS(030520)20200319170303.pdf"),
        ("EKI-2525MI-BE", "../src/sagemaker/data/EKI-2525MI-BE/EKI-2525MI_SI_DS(030520)20200319170234.pdf"),
        ("EKI-2525S-AE", "../src/sagemaker/data/EKI-2525S-AE/EKI-2525M_S_DS(06.19.17)20170705202136.pdf"),
        ("EKI-2728MI-BE", "../src/sagemaker/data/EKI-2728MI-BE/EKI-2728M_MI-_-S_SI_DS(112117)2017112416094120200212175644.pdf"),
        ("EKI-2741F-BE", "../src/sagemaker/data/EKI-2741F-BE/EKI-2741F-BE_DS(10.18.17)20181121142215.pdf"),
        ("EKI-5525I-AE", "../src/sagemaker/data/EKI-5525I-AE/EKI-5525I20150714145223.pdf"),
        ("EKI-2741LX-BE", "../src/sagemaker/data/EKI-2741LX-BE/EKI-2741F-BE_DS(10.18.17)20181121142352.pdf"),
        ("EKI-2528I-BE", "../src/sagemaker/data/EKI-2528I-BE/EKI-2525_I_2528_I_DS(110320)20201103184942.pdf"),
        ("EKI-2705G-1GPI-A", "../src/sagemaker/data/EKI-2705G-1GPI-A/EKI-2705G-1GPI_2705E-1GPI_DS(082624)20240826164151.pdf"),
        ("EKI-2528I-LA-AE", "../src/sagemaker/data/EKI-2528I-LA-AE/EKI-2528I-LA_DS(052719)20241021111145.pdf"),
        ("EKI-5729FI-AE", "../src/sagemaker/data/EKI-5729FI-AE/EKI-5729F_FI_DS(082120)20200829165142.pdf"),
        ("EKI-2720G-4FI-AE", "../src/sagemaker/data/EKI-2720G-4FI-AE/EKI-2720G-4F_4FI_DS(11.06.18)20190605150656.pdf"),
        ("EKI-2528-BE", "../src/sagemaker/data/EKI-2528-BE/EKI-2525_I_2528_I_DS(110320)20201103184914.pdf"),
        ("EKI-2526S-AE", "../src/sagemaker/data/EKI-2526S-AE/DS_EKI-2525M_2526M_S20150115170853.pdf"),
        ("EKI-2741FHPI-AE", "../src/sagemaker/data/EKI-2741FHPI-AE/EKI-2741FHPI_DS(12.22.17)20171225173911.pdf"),
        ("EKI-2525M-BE", "../src/sagemaker/data/EKI-2525M-BE/EKI-2525M_S_DS(061917)20200420184011.pdf"),
        ("EKI-2528-C", "../src/sagemaker/data/EKI-2528-C/EKI-2525_I_2528_I_DS(092723)20230928103744.pdf"),
        # 原45個案例中的25個（續）
        ("EKI-2428G-4CA-AE", "../src/sagemaker/data/EKI-2428G-4CA-AE/EKI-2428G-4CA_DS(021120)20200317192322.pdf"),
        ("EKI-2525-BE", "../src/sagemaker/data/EKI-2525-BE/EKI-2525_I_2528_I_DS(110320)20201103184806.pdf"),
        ("EKI-2525I-BE", "../src/sagemaker/data/EKI-2525I-BE/EKI-2525_I_2528_I_DS(110320)20201103184839.pdf"),
        ("EKI-2525I-LA-AE", "../src/sagemaker/data/EKI-2525I-LA-AE/EKI-2525I-LA_DS(052719)20240424142128.pdf"),
        ("EKI-2526M-AE", "../src/sagemaker/data/EKI-2526M-AE/DS_EKI-2525M_2526M_S20150115170805.pdf"),
        ("EKI-2528I-M12-AE", "../src/sagemaker/data/EKI-2528I-M12-AE/EKI-2528I-M12_DS(071520)20200715142239.pdf"),
        ("EKI-2701MPI-R-AE", "../src/sagemaker/data/EKI-2701MPI-R-AE/EKI-2701MPI-R_DS(10.12.18)20181026174717.pdf"),
        ("EKI-2705E-1GPI-A", "../src/sagemaker/data/EKI-2705E-1GPI-A/EKI-2705G-1GPI_2705E-1GPI_DS(082624)20240826164212.pdf"),
        ("EKI-2706E-1GFPI-BE", "../src/sagemaker/data/EKI-2706E-1GFPI-BE/EKI-2706G-1GFPI-BE_2706E-1GFPI-BE_DS(082624)20240826164125.pdf"),
        ("EKI-2706G-1GFPI-BE", "../src/sagemaker/data/EKI-2706G-1GFPI-BE/EKI-2706G-1GFPI-BE_2706E-1GFPI-BE_DS(082624)20240826164101.pdf"),
        ("EKI-2720G-4F-AE", "../src/sagemaker/data/EKI-2720G-4F-AE/EKI-2720G-4F_4FI_DS(11.06.18)20190605150656.pdf"),
        ("EKI-2725-CE", "../src/sagemaker/data/EKI-2725-CE/EKI-2725_I_DS(082924)20240829164001.pdf"),
        ("EKI-2725F-AE", "../src/sagemaker/data/EKI-2725F-AE/EKI-2725F_FI_DS(110320)20201103185231.pdf"),
        ("EKI-2725I-CE", "../src/sagemaker/data/EKI-2725I-CE/EKI-2725_I_DS(082924)20240829163932.pdf"),
        ("EKI-2728M-BE", "../src/sagemaker/data/EKI-2728M-BE/EKI-2728M_MI _ S_SI_DS(11.21.17)20171124155954.pdf"),
        ("EKI-2741FI-BE", "../src/sagemaker/data/EKI-2741FI-BE/EKI-2741F-BE_DS(10.18.17)20181121142249.pdf"),
        ("EKI-2741SX-BE", "../src/sagemaker/data/EKI-2741SX-BE/EKI-2741F-BE_DS(10.18.17)20181121142320.pdf"),
        ("EKI-5526I-AE", "../src/sagemaker/data/EKI-5526I-AE/EKI-5526_I_DS(081621)20210823001401.pdf"),
        ("EKI-5526I-PN-AE", "../src/sagemaker/data/EKI-5526I-PN-AE/EKI-5526_I-PN_5528_I-PN_DS(022123)20230221183554.pdf"),
        ("EKI-5528I-PN-AE", "../src/sagemaker/data/EKI-5528I-PN-AE/EKI-5526_I-PN_5528_I-PN_DS(022123)20230221183723.pdf"),
        ("EKI-5626CI-AE", "../src/sagemaker/data/EKI-5626CI-AE/EKI-5626CI_5629CI_DS(03.05.18)20180305175439.pdf"),
        ("EKI-5629CI-AE", "../src/sagemaker/data/EKI-5629CI-AE/EKI-5626CI_5629CI_DS(03.05.18)20180305175341.pdf"),
        ("EKI-5726I-AE", "../src/sagemaker/data/EKI-5726I-AE/EKI-5726NI-A_DS(041724)20240418170505.pdf"),
        ("EKI-5728", "../src/sagemaker/data/EKI-5728/EKI-5725_I_5728_I_DS(082120)20200829165056.pdf"),
        ("EKI-5729PI-AE", "../src/sagemaker/data/EKI-5729PI-AE/EKI-5729P_PI_DS(11.13.18)20181121142646.pdf"),
        
        # 新增的7個PDF測試案例
        ("EKI-2541M-BE", "../src/sagemaker/data/EKI-2541M-BE.pdf"),
        ("EKI-2541S-BE", "../src/sagemaker/data/EKI-2541S-BE.pdf"),
        ("EKI-2541SI-BE", "../src/sagemaker/data/EKI-2541SI-BE.pdf"),
        ("EKI-2701HPI-AE", "../src/sagemaker/data/EKI-2701HPI-AE.pdf"),
        ("EKI-2701PSI-AE", "../src/sagemaker/data/EKI-2701PSI-AE.pdf"),
        ("EKI-2728-D", "../src/sagemaker/data/EKI-2728-D.pdf"),
        ("EKI-2728I-D", "../src/sagemaker/data/EKI-2728I-D.pdf"),
        # 注意：EKI-5626CI-EI-AE 和 EKI-5626CI-MB-AE 使用目錄中的PDF文件
        ("EKI-5626CI-EI-AE", "../src/sagemaker/data/EKI-5626CI-EI-AE/EKI-5626CI-MB-Series_DS(031521)20210504130824.PDF"),
        ("EKI-5626CI-MB-AE", "../src/sagemaker/data/EKI-5626CI-MB-AE/EKI-5626CI-MB-Series_DS(031521)20210504130824.PDF"),
    ]
    
    results = []
    correct_count = 0
    total_count = 0
    mouser_direct_count = 0
    
    start_time = time.time()
    
    for product_model, relative_path in test_cases:
        if product_model not in ground_truth:
            print(f"跳過 {product_model}: 沒有ground truth")
            continue
            
        pdf_path = Path(relative_path)
        if not pdf_path.exists():
            print(f"跳過 {product_model}: PDF文件不存在 {pdf_path}")
            continue
            
        expected_eccn = ground_truth[product_model]
        is_correct, predicted, method, proc_time = test_single_product(
            product_model, pdf_path, expected_eccn
        )
        
        results.append({
            'product_model': product_model,
            'expected_eccn': expected_eccn,
            'predicted_eccn': predicted,
            'method': method,
            'processing_time': proc_time,
            'is_correct': is_correct
        })
        
        if is_correct:
            correct_count += 1
        if method == 'mouser_api_direct':
            mouser_direct_count += 1
        total_count += 1
        
        # 小延遲避免API限制
        time.sleep(1)
    
    total_time = time.time() - start_time
    
    # 統計結果
    print(f"\n 測試結果統計:")
    print(f"  總測試數: {total_count}")
    print(f"  正確數: {correct_count}")
    print(f"  準確率: {correct_count/total_count*100:.1f}%" if total_count > 0 else "N/A")
    print(f"  Mouser直接查詢: {mouser_direct_count} ({mouser_direct_count/total_count*100:.1f}%)")
    print(f"  總耗時: {total_time:.1f}秒")
    print(f"  平均耗時: {total_time/total_count:.2f}秒/個" if total_count > 0 else "N/A")
    
    # 方法統計
    method_stats = {}
    for result in results:
        if result['method']:
            method = result['method']
            if method not in method_stats:
                method_stats[method] = {'total': 0, 'correct': 0}
            method_stats[method]['total'] += 1
            if result['is_correct']:
                method_stats[method]['correct'] += 1
    
    print(f"\n 按方法統計:")
    for method, stats in method_stats.items():
        acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f" {method}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # 顯示詳細結果
    print(f"\n 詳細結果:")
    for result in results:
        status = "" if result['is_correct'] else ""
        print(f" {status} {result['product_model']}: {result['predicted_eccn']} (期望: {result['expected_eccn']}) - {result['method']}")
    
    # 錯誤案例分析
    errors = [r for r in results if not r['is_correct']]
    if errors:
        print(f"\n 錯誤案例分析:")
        for error in errors:
            print(f"  {error['product_model']}: 預測 {error['predicted_eccn']} vs 期望 {error['expected_eccn']}")
    
    # 保存結果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"complete_hardcoded_test_results_{timestamp}.json"
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_count,
        'correct_count': correct_count,
        'accuracy_rate': correct_count/total_count*100 if total_count > 0 else 0,
        'mouser_direct_count': mouser_direct_count,
        'mouser_usage_rate': mouser_direct_count/total_count*100 if total_count > 0 else 0,
        'total_time': total_time,
        'method_stats': method_stats,
        'detailed_results': results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 結果已保存到: {results_file}")

if __name__ == "__main__":
    main()