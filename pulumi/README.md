# ECCN Complete Pipeline - 必要檔案

##核心Pipeline檔案
- `lambda_function.py` - 完整Pipeline主處理函數（Main Classifier）
- `pdf_parser_with_requests.py` - PDF Parser Lambda函數
- `tools.py` - Mouser API整合和工具函數
- `websearch.py` - WebSearch交叉驗證模組
- `prompts.py` - System prompts
- `data.pkl` - ECCN embeddings資料

##部署檔案
- `complete-pipeline-function.zip` - Main Classifier部署包
- `pdf-parser-function.zip` - PDF Parser部署包
- `requests_layer.zip` - PDF Parser需要的Lambda layer
- `Pulumi.yaml` - AWS基礎設施配置

##Pipeline流程
1. 優先Mouser API直接查詢 → 找到則直接返回
2. 查詢不到 → 同時執行: PDF特徵→Mouser相似產品查詢 + WebSearch交叉驗證  
3. 將所有結果給LLM綜合決策
4. 顯示完整資料來源
5. 一個curl完成整套流程

##使用方式
```bash
curl -X POST https://c22tjivksajimzodkbqfz4zdly0wcirj.lambda-url.us-east-1.on.aws/ \
  -H "Content-Type: application/json" \
  -d '{
    "s3_key": "parsed/pdf_file.json",
    "product_model": "PRODUCT_MODEL",
    "debug": true
  }'
```
