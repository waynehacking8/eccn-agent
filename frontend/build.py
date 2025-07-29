#!/usr/bin/env python3
"""
Cloudflare Pages 建置腳本
為生產環境準備Gradio應用
"""

import os
import shutil
import subprocess
import sys

def create_static_build():
    """為Cloudflare Pages創建靜態建置"""
    print("🏗️ 建置ECCN智能分析系統用於Cloudflare Pages...")
    
    # 創建build目錄
    build_dir = "dist"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    os.makedirs(build_dir)
    
    # 複製必要文件
    files_to_copy = [
        "index.html"
    ]
    
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, build_dir)
            print(f"✅ 複製 {file}")
    
    # 創建_headers文件用於CORS
    headers_content = """/*
  Access-Control-Allow-Origin: *
  Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
  Access-Control-Allow-Headers: Content-Type, Authorization
  X-Frame-Options: SAMEORIGIN
  X-Content-Type-Options: nosniff
"""
    
    with open(os.path.join(build_dir, "_headers"), "w") as f:
        f.write(headers_content)
    
    # 創建_redirects文件
    redirects_content = """/api/proxy https://uk77kivopn5ivjsjjyci4uewha0erpwb.lambda-url.us-east-1.on.aws/ 200
/api/parse/* https://uk77kivopn5ivjsjjyci4uewha0erpwb.lambda-url.us-east-1.on.aws/:splat 200
/api/classify/* https://svls3lp6ulqwastjdrs3snpfce0cwazp.lambda-url.us-east-1.on.aws/:splat 200
"""
    
    with open(os.path.join(build_dir, "_redirects"), "w") as f:
        f.write(redirects_content)
    
    print(f"✅ 建置完成於 {build_dir} 目錄")
    return build_dir

def setup_environment():
    """設定環境變數"""
    env_vars = {
        "ENVIRONMENT": "production",
        "PORT": "8080",
        "DEBUG": "false"
    }
    
    print("🔧 設定環境變數...")
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key}={value}")

def install_dependencies():
    """跳過Python依賴安裝（使用靜態HTML）"""
    print("📦 跳過Python依賴安裝（使用靜態HTML介面）")
    print("✅ 無需安裝依賴")
    return True

def main():
    """主建置流程"""
    print("🚀 開始建置ECCN智能分析系統...")
    
    # 安裝依賴
    if not install_dependencies():
        sys.exit(1)
    
    # 設定環境
    setup_environment()
    
    # 創建建置
    build_dir = create_static_build()
    
    print("\n🎉 建置完成！")
    print(f"📂 建置文件位於: {build_dir}")
    print("\n📋 接下來的步驟:")
    print("1. 安裝 Wrangler CLI: npm install -g wrangler")
    print("2. 登入 Cloudflare: wrangler login")
    print("3. 部署應用: wrangler pages deploy dist --project-name eccn-frontend")
    
if __name__ == "__main__":
    main()