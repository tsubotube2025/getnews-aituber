import asyncio
import json
import websockets
import os
import random
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from google.api_core import exceptions as google_exceptions

load_dotenv()

# ================= 設定エリア =================
# ★マルチAPIキー読み込み
API_KEYS = []
if os.getenv("GOOGLE_API_KEY"): API_KEYS.append(os.getenv("GOOGLE_API_KEY"))
if os.getenv("GOOGLE_API_KEY_2"): API_KEYS.append(os.getenv("GOOGLE_API_KEY_2"))

if not API_KEYS:
    print("エラー: .envにGOOGLE_API_KEYが設定されていません")
    exit()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 

SERVER_HOST = "localhost"
SERVER_PORT = 8000
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/direct-speech"
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"

# 検索対象ドメイン
BASE_DOMAINS = [] 
ADDITIONAL_SITES = [
    "gaitame.com",
    "zai.diamond.jp",
    "fx.minkabu.jp",
    "x.com",
    "twitter.com"
]
ALL_TARGET_DOMAINS = BASE_DOMAINS + ADDITIONAL_SITES

# ★ここに復活させました（日付は自動で付きます）
SEARCH_QUERY_BASE = "為替 FX 市場ニュース 最新 ドル円 ユーロドル ポンド 中銀総裁 連銀総裁 日銀総裁 Min_FX MktBrain Yuto_Headline"

key_index = 0
# ============================================

def get_next_api_key():
    global key_index
    api_key = API_KEYS[key_index]
    key_index = (key_index + 1) % len(API_KEYS)
    return api_key

async def fetch_news_tavily(query):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tavilyで検索中...")
    # Tavilyの設定
    tavily = TavilySearchResults(
        max_results=5, include_answer=False, include_raw_content=True, 
        include_domains=ALL_TARGET_DOMAINS,
        topic="news", days=1
    )
    try:
        return tavily.invoke({"query": query})
    except Exception as e:
        print(f"Tavilyエラー: {e}")
        return None

async def process_with_gemini(tavily_results):
    if not tavily_results: return None
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    context_text = ""
    for item in tavily_results:
        context_text += f"URL: {item['url']}\n本文: {item.get('content', '')}\n---\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        あなたはAITuber「ミュー」だ。現在時刻は {current_time} だ。
        検索結果から**「現在時刻から1時間以内」**に配信された最新の為替ニュースを探せ。
        
        【厳格な判定ルール】
        1. **時間厳守:** 記事内の日時表記を必ず確認し、数時間前や昨日の古い情報は無視しろ。
        2. **なしの場合:** 直近1時間以内の情報がなければ "NO_NEWS" とだけ返せ。
        3. **形式:** JSON形式: {{{{ "type": "chat", "text": "（ミューのセリフ80文字以内）" }}}}
        """),
        ("human", "【検索結果】\n{context_text}")
    ])
    
    # リトライ & キーローテーション
    max_retries = 3
    for attempt in range(max_retries):
        try:
            current_api_key = get_next_api_key()
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_NAME, temperature=0.7, google_api_key=current_api_key
            )
            chain = prompt | llm
            response = await chain.ainvoke({"context_text": context_text[:20000]})
            content = response.content.strip()
            
            if content.startswith("```"): content = content.replace("```json", "").replace("```", "").strip()
            if "NO_NEWS" in content: return None
            return content

        except google_exceptions.ResourceExhausted:
            print(f"⚠️ キー制限(429)。次のキーへ切り替え ({attempt+1}/{max_retries})")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"Geminiエラー: {e}")
            return None
    return None

async def send_oneshot_message(message_data):
    try:
        if isinstance(message_data, str):
            try: payload = json.loads(message_data)
            except: payload = {"type": "chat", "text": message_data}
        else: payload = message_data
        
        if "type" not in payload: payload = {"type": "chat", "text": payload.get("text", str(payload))}
        
        json_str = json.dumps(payload, ensure_ascii=False)
        print(f"送信中...: {json_str}")
        
        async with websockets.connect(WS_URL) as websocket:
            await websocket.send(json_str)
            print(">> 送信完了")

    except ConnectionRefusedError:
        print(f"送信失敗: MT5アプリ(ポート{SERVER_PORT})が起動していません。")
    except Exception as e:
        print(f"送信エラー: {e}")

async def main():
    print(f"=== ミュー (特化検索モード) ===")
    print(f"登録キー数: {len(API_KEYS)}")
    print("------------------------------------------------")

    await send_oneshot_message(f"接続確認！キー{len(API_KEYS)}本体制で重要人物の発言も監視するぞ！")

    while True:
        today_str = datetime.now().strftime('%Y-%m-%d')
        
        # ★ここで日付と結合して最終的なクエリを作ります
        current_query = f"{SEARCH_QUERY_BASE} {today_str}"

        # ニュース検索
        raw = await fetch_news_tavily(current_query)
        if raw:
            json_res = await process_with_gemini(raw)
            if json_res:
                await send_oneshot_message(json_res)
            else:
                print(">> ニュースなし (NO_NEWS)")
        
        print("次回検索まで5分待機...")
        for _ in range(300):
            await asyncio.sleep(1)

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main())
    except KeyboardInterrupt: print("終了")