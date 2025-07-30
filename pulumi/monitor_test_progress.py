#!/usr/bin/env python3
"""
監控測試進度
"""
import time
import os
import subprocess
import json
from datetime import datetime

def check_test_status():
    """檢查測試狀態"""
    try:
        # 檢查進程是否還在運行
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        
        if 'complete_hardcoded_test.py' in result.stdout:
            print(f"測試進程正在運行中... {datetime.now().strftime('%H:%M:%S')}")
            
            # 檢查是否有新的結果文件
            json_files = [f for f in os.listdir('.') if f.startswith('complete_hardcoded_test_results_') and f.endswith('.json')]
            if json_files:
                latest_file = max(json_files, key=os.path.getctime)
                print(f"最新結果文件: {latest_file}")
                
                # 如果文件是最近創建的，顯示內容
                if os.path.getctime(latest_file) > time.time() - 300:  # 5分鐘內
                    try:
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            total = data.get('total_tests', 0)
                            correct = data.get('correct_count', 0)
                            accuracy = data.get('accuracy_rate', 0)
                            print(f"當前結果: {correct}/{total} ({accuracy:.1f}%)")
                    except:
                        print("結果文件還在寫入中...")
            
            return True
        else:
            print(f"測試進程已結束 {datetime.now().strftime('%H:%M:%S')}")
            
            # 檢查是否有最終結果
            json_files = [f for f in os.listdir('.') if f.startswith('complete_hardcoded_test_results_') and f.endswith('.json')]
            if json_files:
                latest_file = max(json_files, key=os.path.getctime)
                print(f"最終結果文件: {latest_file}")
                
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        total = data.get('total_tests', 0)
                        correct = data.get('correct_count', 0)
                        accuracy = data.get('accuracy_rate', 0)
                        total_time = data.get('total_time', 0)
                        print(f"最終結果: {correct}/{total} ({accuracy:.1f}%) - 耗時: {total_time:.1f}s")
                        
                        # 顯示方法統計
                        method_stats = data.get('method_stats', {})
                        print(f"方法統計:")
                        for method, stats in method_stats.items():
                            acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                            print(f"  {method}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
                except Exception as e:
                    print(f"讀取結果文件失敗: {e}")
            
            return False
            
    except Exception as e:
        print(f"檢查狀態失敗: {e}")
        return False

def main():
    """主函數"""
    print("開始監控測試進度...")
    print("按 Ctrl+C 停止監控\n")
    
    try:
        while True:
            if not check_test_status():
                print("測試已完成或進程不存在")
                break
            
            print("-" * 50)
            time.sleep(30)  # 每30秒檢查一次
            
    except KeyboardInterrupt:
        print("\n監控已停止")

if __name__ == "__main__":
    main()