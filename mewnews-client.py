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

# ★重要: ここは既存アプリが動いているアドレスです
SERVER_HOST = "localhost"
SERVER_PORT = 9000
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/direct-speech"

GEMINI_MODEL_NAME = "gemini-2.0-flash-exp"

# 基本のターゲットドメイン + 追加サイト
BASE_DOMAINS = ["bloomberg.co.jp", "reuters.com", "investing.com", "fxstreet.jp", "min-fx.jp", "nikkei.com"]
ADDITIONAL_SITES = ["gaitame.com", "zai.diamond.jp"] # 必要に応じて追加
ALL_TARGET_DOMAINS = BASE_DOMAINS + ADDITIONAL_SITES

SEARCH_QUERY = "為替 FX 市場ニュース 最新 ドル円 ユーロドル"
# ============================================

async def fetch_news_tavily(query):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tavilyで検索中...")
    tavily = TavilySearchResults(
        max_results=5,
        include_answer=False,
        include_raw_content=True, 
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

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
        あなたはAITuber「ミュー」だ。現在時刻は {current_time} だ。
        検索結果から**「現在時刻から1時間以内」**に配信された最新の為替ニュースを探せ。
        
        【厳格な判定ルール】
        1. **時間厳守:** 記事内の日時表記を必ず確認し、数時間前や昨日の古い情報は無視しろ。
        2. **なしの場合:** 直近1時間以内の情報がなければ "NO_NEWS" とだけ返せ。
        
        【発言のルール】
        1. **ソース名は言わない:** サイト名やURLは読み上げず、自分の言葉として話すこと。
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

async def main():
    print(f"=== ミュー (クライアントモード) ===")
    print(f"接続先: {WS_URL}")
    print("既存のアプリ(9000番)へニュースを送信します。")
    print("------------------------------------------------")

    # 再接続ループ
    while True:
        try:
            # ★変更点: serveではなくconnectを使う
            async with websockets.connect(WS_URL) as websocket:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 既存アプリに接続成功！")
                
                # 接続時の挨拶
                await websocket.send(json.dumps({"type": "chat", "text": "接続したぞ！ここからニュースを送るからな！"}, ensure_ascii=False))
                
                # ニュース監視ループ
                while True:
                    # 検索実行
                    raw = await fetch_news_tavily(SEARCH_QUERY)
                    if raw:
                        json_res = await process_with_gemini(raw)
                        if json_res:
                            # 既存アプリへ送信
                            if isinstance(json_res, str):
                                try: payload = json.loads(json_res)
                                except: payload = {"type": "chat", "text": json_res}
                            else: payload = json_res
                            
                            if "type" not in payload: payload = {"type": "chat", "text": payload.get("text", str(payload))}
                            
                            msg = json.dumps(payload, ensure_ascii=False)
                            print(f"送信: {msg}")
                            await websocket.send(msg)
                        else:
                            print(">> ニュースなし (NO_NEWS)")
                    
                    print("次回検索まで5分待機...")
                    await asyncio.sleep(300)

        except (ConnectionRefusedError, OSError) as e:
            print(f"接続エラー: 既存アプリ(ポート9000)が起動していないようです。再試行中... ({e})")
            await asyncio.sleep(5)
        except websockets.ConnectionClosed:
            print("切断されました。再接続します...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"予期せぬエラー: {e}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main())
    except KeyboardInterrupt: print("終了")