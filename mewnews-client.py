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

# ★変更点1: ポートを8000に変更 (MT5アプリに合わせる)
SERVER_HOST = "localhost"
SERVER_PORT = 8000
WS_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/direct-speech"

# ★変更点2: モデルをFlash Liteに変更
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"

# 検索対象ドメイン (大手ニュース + 追加サイト)
BASE_DOMAINS = [
   # "bloomberg.co.jp",
   # "reuters.com",
    #"investing.com",
    #"fxstreet.jp",
   # "min-fx.jp",
    #"nikkei.com"

]
ADDITIONAL_SITES = [
    "gaitame.com",
    "zai.diamond.jp",
    "https://fx.minkabu.jp/news",
    "https://min-fx.jp/market/news/",
    "https://x.com/Min_FX",
    "https://x.com/MktBrain",
    "https://x.com/Yuto_Headline"



]
ALL_TARGET_DOMAINS = BASE_DOMAINS + ADDITIONAL_SITES

SEARCH_QUERY = "為替 FX 市場ニュース 最新 ドル円 ユーロドル ポンド"
# ============================================

async def fetch_news_tavily(query):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tavilyで検索中... (Query: {query})")
    
    tavily = TavilySearchResults(
        max_results=5,
        include_answer=False,
        include_raw_content=True, 
        include_domains=ALL_TARGET_DOMAINS,
        # ★重要変更点: search_depth は advanced のままでOKですが、
        # TavilyのAPIパラメータを追加して最新ニュースに絞ります
        # (LangChainのバージョンによっては kwargs で渡す必要があります)
        topic="news",  # ニュースモード指定
        days=1         # 直近1日（24時間）以内の記事に限定
    )
    
    try:
        # 検索実行
        # ※毎回クエリに最新の日付を入れるため、引数で渡されたqueryを使います
        # ただし、mainループ側で search_query を更新する必要があります
        return tavily.invoke({"query": query})
    except Exception as e:
        print(f"Tavilyエラー: {e}")
        return None

async def process_with_gemini(tavily_results):
    if not tavily_results: return None
    
    # Gemini 2.0 Flash Lite で初期化
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL_NAME,
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )
    
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
        1. **ソース名は言わない:** サイト名やURLは読み上げず、自分の言葉として話すこと。〇〇によると等も禁止。
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

# ★変更点3: 送信関数の中に connect を移動 (一撃離脱方式)
async def send_oneshot_message(message_data):
    """
    MT5アプリ(ポート8000)に一瞬だけ接続してメッセージを送り、
    即座に切断する。これによりMT5側の負荷や音声遅延を防ぐ。
    """
    try:
        # JSONデータの整形
        if isinstance(message_data, str):
            try: payload = json.loads(message_data)
            except: payload = {"type": "chat", "text": message_data}
        else:
            payload = message_data
        
        if "type" not in payload:
            payload = {"type": "chat", "text": payload.get("text", str(payload))}
        
        json_str = json.dumps(payload, ensure_ascii=False)
        print(f"送信中...: {json_str}")
        
        # ★ここで接続 → 送信 → 自動切断
        async with websockets.connect(WS_URL) as websocket:
            await websocket.send(json_str)
            print(">> 送信完了（切断しました）")

    except ConnectionRefusedError:
        print(f"送信失敗: MT5アプリ(ポート{SERVER_PORT})が起動していません。")
    except Exception as e:
        print(f"送信エラー: {e}")

async def main():
    print(f"=== ミュー (MT5連携・軽量版) ===")
    print(f"モデル: {GEMINI_MODEL_NAME}")
    print(f"送信先: {WS_URL}")
    print("------------------------------------------------")

    # 1. 起動時の挨拶
    await send_oneshot_message("ニュースエージェント、接続確認よし！監視を開始するぞ！")

    # 2. ニュース監視ループ
    while True:
          # ★ループのたびに日付入りクエリを作成
        today_str = datetime.now().strftime('%Y-%m-%d')
        current_query = f"為替 FX 市場ニュース 最新 ドル円 ユーロドル ポンド {today_str}"
        # ニュース検索
        raw = await fetch_news_tavily(current_query)
        if raw:
            json_res = await process_with_gemini(raw)
            if json_res:
                # ニュースがある時だけ接続して送る
                await send_oneshot_message(json_res)
            else:
                print(">> ニュースなし (NO_NEWS)")
        
        print("次回検索まで5分待機...")
        await asyncio.sleep(90)

if __name__ == "__main__":
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try: asyncio.run(main())
    except KeyboardInterrupt: print("終了")