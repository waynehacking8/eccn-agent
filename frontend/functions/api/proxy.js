// Cloudflare Pages Functions API代理
// 將API請求代理到AWS Lambda

export async function onRequestPost(context) {
  const { request } = context;
  
  try {
    const url = new URL(request.url);
    const path = url.pathname;
    
    // 根據路徑決定目標Lambda URL
    let targetUrl;
    if (path.includes('/api/parse') || path.includes('/parse')) {
      // PDF Parser Lambda
      targetUrl = 'https://uk77kivopn5ivjsjjyci4uewha0erpwb.lambda-url.us-east-1.on.aws/';
    } else if (path.includes('/api/classify') || path.includes('/classify')) {
      // Main Classifier Lambda  
      targetUrl = 'https://svls3lp6ulqwastjdrs3snpfce0cwazp.lambda-url.us-east-1.on.aws/';
    } else {
      // 預設使用PDF Parser（單一端點處理完整流程）
      targetUrl = 'https://uk77kivopn5ivjsjjyci4uewha0erpwb.lambda-url.us-east-1.on.aws/';
    }
    
    // 代理請求到AWS Lambda
    const lambdaResponse = await fetch(targetUrl, {
      method: request.method,
      headers: {
        'Content-Type': request.headers.get('Content-Type'),
        'User-Agent': 'Cloudflare-Pages-Proxy/1.0',
      },
      body: request.body,
    });
    
    // 複製回應
    const responseBody = await lambdaResponse.text();
    
    return new Response(responseBody, {
      status: lambdaResponse.status,
      statusText: lambdaResponse.statusText,
      headers: {
        'Content-Type': lambdaResponse.headers.get('Content-Type') || 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      },
    });
    
  } catch (error) {
    console.error('API代理錯誤:', error);
    
    return new Response(JSON.stringify({
      success: false,
      error: '服務暫時無法使用，請稍後再試',
      details: error.message
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
      },
    });
  }
}

// 處理GET請求（用於健康檢查）
export async function onRequestGet(context) {
  return new Response(JSON.stringify({
    status: 'ok',
    message: 'ECCN Gradio Frontend API Proxy',
    timestamp: new Date().toISOString(),
    endpoints: {
      parse: '/api/proxy (POST with PDF)',
      classify: '/api/proxy (POST with JSON)'
    }
  }), {
    headers: {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
    },
  });
}