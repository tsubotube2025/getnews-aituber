import asyncio
import json
import websockets
import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# ================= 設定エリア =================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 

SERVER_HOST = "localhost"
SERVER_PORT = 9000
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/direct-speech"
GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

# 1. 基本のターゲットドメイン（大手ニュース）
BASE_DOMAINS = [
    "bloomberg.co.jp",
    "reuters.com",
    "investing.com",
    "fxstreet.jp",
    "min-fx.jp",
    "nikkei.com"
]

# 2. ★ここに追加したいWebサイトのドメインを記述してください★
# 例: "gaitame.com", "diamond.jp" など
ADDITIONAL_SITES = [
    "gaitame.com",
    "zai.diamond.jp",
    # ここに自由に足せます
]

# 検索対象を合体（このリスト内のサイトだけを検索します）
ALL_TARGET_DOMAINS = BASE_DOMAINS + ADDITIONAL_SITES

# 検索クエリ
SEARCH_QUERY = "為替 FX 市場ニュース 最新 ドル円 ユーロドル"

connected_clients = set()
# ============================================

async def fetch_news_tavily(query):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 指定された {len(ALL_TARGET_DOMAINS)} サイトから検索中...")
    
    tavily = TavilySearchResults(
        max_results=5,
        include_answer=False,
        include_raw_content=True, 
        # ★ここで検索範囲を「指定したリスト」だけに限定します
        include_domains=ALL_TARGET_DOMAINS,
        search_depth="advanced",
    )
    try:
        return tavily.invoke({"query": query})
    except Exception as e:
        print(f"Tavilyエラー: {e}")
        return None

async def process_with_gemini(tavily_results):
    if not tavily_results: return None
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0.7, google_api_key=GOOGLE_API_KEY)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    context_text = ""
    for item in tavily_results:
        context_text += f"URL: {item['url']}\n本文: {item.get('content', '')}\n---\n"

    # プロンプト設定
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        あなたはAITuber「ミュー」だ。現在時刻は {current_time} だ。
        検索結果から**「現在時刻から1時間以内」**に配信された最新の為替ニュースを探せ。
        
        【厳格な判定ルール】
        1. **時間厳守:** 記事内の日時表記を必ず確認し、数時間前や昨日の古い情報は無視しろ。
        2. **なしの場合:** 直近1時間以内の情報がなければ "NO_NEWS" とだけ返せ。
        
        【発言のルール】
        1. **ソース名は言わない:** 情報源（サイト名）やURLは読み上げないこと。中身だけを自分の言葉として話すこと。
        2. **キャラ設定:** 語尾は「〜だ！」「〜らしいな！」など元気よく。
        3. **形式:** JSON形式: {{{{ "type": "chat", "text": "（ミューのセリフ80文字以内）" }}}}
        """),
        ("human", "【検索結果】\n{context_text}")
    ])
    
    try:
        chain = prompt | llm
        response = await chain.ainvoke({"context_text": context_text[:20000]})
        content = response.content.strip()
        if content.startswith("```"): content = content.replace("```json", "").replace("```", "").strip()
        if "NO_NEWS" in content: return None
        return content
    except Exception as e:
        print(f"Geminiエラー: {e}")
        return None

async def broadcast(message_data):
    if not connected_clients: return
    if isinstance(message_data, str):
        try: payload = json.loads(message_data)
        except: payload = {"type": "chat", "text": message_data}
    else: payload = message_data
    if "type" not in payload: payload = {"type": "chat", "text": payload.get("text", str(payload))}
    
    json_str = json.dumps(payload, ensure_ascii=False)
    print(f"送信: {json_str}")
    for ws in connected_clients.copy():
        try: await ws.send(json_str)
        except: connected_clients.remove(ws)

async def connection_handler(websocket):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ★接続成功！★")
    connected_clients.add(websocket)
    try:
        await websocket.send(json.dumps({"type": "chat", "text": "接続完了！指定されたサイトを監視するぞ！"}, ensure_ascii=False))
        await websocket.wait_closed()
    finally:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 切断したぜ！(ブラウザが閉じられました)")
        connected_clients.remove(websocket)

async def news_loop():
    while len(connected_clients) == 0: await asyncio.sleep(1)
    print("検索開始...")
    while True:
        raw = await fetch_news_tavily(SEARCH_QUERY)
        if raw:
            json_res = await process_with_gemini(raw)
            if json_res: await broadcast(json_res)
            else: print(">> 1時間以内のニュースなし (NO_NEWS)")
        await asyncio.sleep(300)

async def main():
    print(f"=== Tavily版 (指定ドメイン限定モード) 起動 ===")
    server = await websockets.serve(connection_handler, SERVER_HOST, SERVER_PORT)
    await asyncio.gather(server.wait_closed(), news_loop())

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main())
    except KeyboardInterrupt: print("終了")