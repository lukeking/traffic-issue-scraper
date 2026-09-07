"""
analyzer.py
使用 Gemini REST API 對每篇新聞進行摘要 + 深度分析
支援兩種內容類型：traffic（台灣機車交通）和 ffxiv（FFXIV 遊戲資訊）
（直接用 requests，不依賴 google-generativeai SDK，相容 Python 3.8+）
"""

import os
import re
import time
import logging
import random
import requests

logger = logging.getLogger(__name__)

# GitHub Actions 若設定 GEMINI_MODEL_NAME secret 但留空，get 會得到 "" 而非預設值
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL_NAME") or "").strip() or "gemini-2.5-pro"
logger.debug("使用 Gemini 模型：%s", GEMINI_MODEL)
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={api_key}"
)
GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-embedding-2:embedContent?key={api_key}"
)
EMBED_DIMENSIONS = 768  # MRL truncation from 3072; balances accuracy vs storage

SYSTEM_PROMPT = (
    "你是台灣交通新聞分析師，專精機車議題（含白牌普通重型機車、紅黃牌大型重型機車）。"
    "請用繁體中文回應，語氣客觀專業，適合台灣機車騎士閱讀。"
    "回應格式務必嚴格按照指示，不要加入任何額外說明或 markdown 符號。"
)

FFXIV_SYSTEM_PROMPT = (
    "你是 FFXIV（最終幻想XIV）資訊分析師，專精遊戲攻略、版本更新與職業機制。"
    "請用繁體中文回應，語氣客觀專業，適合台灣 FFXIV 玩家閱讀。"
    "回應格式務必嚴格按照指示，不要加入任何額外說明或 markdown 符號。"
    "翻譯遊戲術語時，必須以提供的知識庫為準，不得自行發明未收錄的譯名。"
)

YOUTUBE_SYSTEM_PROMPT = (
    "你是台灣交通議題影片分析師，專精機車與道路安全相關的 YouTube 評論與觀察內容。"
    "你的任務是從影片逐字稿或說明中提取核心論點與比較洞察，而非單純摘要新聞事件。"
    "請用繁體中文回應，語氣客觀專業，適合台灣機車騎士閱讀。"
    "回應格式務必嚴格按照指示，不要加入任何額外說明或 markdown 符號。"
)

YOUTUBE_ANALYSIS_PROMPT_TEMPLATE = """以下是一段 YouTube 影片評論或分析的逐字稿（或說明）：

標題：{title}
頻道：{source}
逐字稿／說明：{summary}
連結：{link}

請依照以下格式回應，每個欄位必須在同一行內完成，不可換行：

摘要：用2到3句話說明這段影片的主要論點或觀察是什麼（同一行寫完，句子之間用「。」分隔）
分析：用4到6句話提取影片的核心比較論點或分析洞察，說明它對台灣機車騎士或交通政策的啟示（同一行寫完，句子之間用「。」分隔）
重要性：高（或中或低，只填一個字）
重要性原因：一句話說明為何如此評定（同一行寫完）
標籤：從以下選項中選2到4個最相關的標籤，用逗號分隔，也可加入未列出但更精確的標籤（同一行寫完）
可選標籤：{available_tags}
地區：填入最相關的地理標籤，如「台北市」「全台」「日本」「歐洲」「不明」（只填一個，同一行寫完）

---
重要性評定標準（請嚴格遵守，預期分布約：高 30%、中 50%、低 20%）：

【高】滿足以下任一條件：
- 提供台灣交通現況的深度比較分析（如對比日本、歐洲法規或設計）
- 揭露系統性道路安全問題，對騎士有直接參考價值
- 影片引發廣泛討論或對政策有實質影響潛力

【中】滿足以下任一條件：
- 觀點有參考價值但範圍較窄（如單一路段或單一事件的評論）
- 資訊來源可信但分析深度有限

【低】滿足以下條件：
- 純娛樂性或與交通安全改善無直接關係
- 內容重複性高、無新見解
"""

DEFAULT_TAGS = [
    "法規", "事故", "停車", "路權", "重機", "電動", "考照",
    "國道", "工程", "Gogoro", "白牌", "紅牌", "黃牌", "取締",
    "新制", "道路設計", "交通安全",
]

FFXIV_DEFAULT_TAGS = [
    "零式", "絕境戰", "職業調整", "新內容", "版本更新",
    "活動", "修正", "地下城", "討伐戰", "同盟突擊",
    "製作採集", "置房", "裝備", "7.x", "蝮蛇師", "繪靈法師",
]

_KB_CACHE: dict | None = None
_KB_KATAKANA = re.compile(r"[ァ-ヺー]{3,}")
_KB_HIGHLIGHT = re.compile(r"\[\[([^\]]+)\]\]")
_KB_MISS_ACCUMULATOR: set = set()

ANALYSIS_PROMPT_TEMPLATE = """以下是一則台灣交通相關新聞：

標題：{title}
來源：{source}
原始摘要：{summary}
連結：{link}

請依照以下格式回應，每個欄位必須在同一行內完成，不可換行：

摘要：用2到3句話說明事件核心是什麼、涉及哪些對象（同一行寫完，句子之間用「。」分隔）
分析：用4到6句話分析背景原因、對機車騎士的實際影響、政策趨勢或值得關注的面向（同一行寫完，句子之間用「。」分隔）
重要性：高（或中或低，只填一個字）
重要性原因：一句話說明為何如此評定（同一行寫完）
標籤：從以下選項中選2到4個最相關的標籤，用逗號分隔，也可加入未列出但更精確的標籤（同一行寫完）
可選標籤：{available_tags}
地區：填入最相關的地理標籤，如「台北市」「新北市」「台中市」「高雄市」「全台」「不明」（只填一個，同一行寫完）

---
重要性評定標準（請嚴格遵守，預期分布約：高 30%、中 50%、低 20%）：

【高】滿足以下任一條件：
- 有人員傷亡（死亡或重傷）的交通事故
- 影響全國或多縣市機車族的法規、政策新制（實際上路或立法院通過）
- 重大道路工程設計缺失，直接威脅騎士安全
- 取締專案或重大違規執法（影響範圍廣、罰則重）

【中】滿足以下任一條件：
- 地方性事故（輕傷或財損為主）
- 政策討論、草案、評估階段（尚未定案）
- 道路工程進度更新、爭議討論（無立即安全威脅）
- 新產品/新服務上市對機車族有間接影響
- 違規取締（一般性、地方性）

【低】滿足以下條件：
- 純報導性、無直接影響（如活動、評比、展覽）
- 重複性新聞（同一事件的後續跟進報導、無新資訊）
- 對機車騎士影響極小或高度不確定
"""


FFXIV_ANALYSIS_TEMPLATE = """以下是一則 FFXIV 相關資訊：

標題：{title}
來源：{source}
原始摘要：{summary}
連結：{link}

【FFXIV 知識庫 — 請嚴格使用以下對照翻譯，不得自行發明未收錄的譯名】
{knowledge_base}

【重要規則：未收錄術語的處理方式】
- 若原文中出現以片假名、平假名或日文漢字書寫的術語，且未收錄於上方知識庫，請保留日文原文並以 [[術語]] 格式包覆（例：[[エーテル]]）。
- 英文術語、英文縮寫（如 CF、FC、MSQ、NPC）及繁體中文術語，即使未出現在知識庫，也不需要用 [[]] 包覆，直接使用即可。
- 絕對不可自行翻譯未收錄的日文術語。

請依照以下格式回應，每個欄位必須在同一行內完成，不可換行：

摘要：用2到3句話說明這則資訊的核心內容（同一行寫完，句子之間用「。」分隔）
分析：用4到6句話分析對玩家的實際影響、版本/職業/內容的重要性、值得關注的面向（同一行寫完）
重要性：高（或中或低，只填一個字）
重要性原因：一句話說明為何如此評定（同一行寫完）
標籤：從以下選項中選2到4個最相關的標籤，用逗號分隔，也可加入未列出但更精確的標籤（同一行寫完）
可選標籤：{available_tags}

---
重要性評定標準（請嚴格遵守）：

【高】任一條件：
- 重大版本更新（零式或絕境戰新內容上線、職業重大改版、新資料片）
- 全服公告性重要變更（伺服器異動、系統重大調整）

【中】任一條件：
- 職業平衡調整、限時活動開始/結束
- 一般版本修正、新增 QoL 功能、新裝備

【低】：
- 次要修正、一般公告、對遊戲玩法無直接影響的資訊
"""


def load_knowledge_base() -> dict:
    """
    從 Supabase knowledge_base 表格載入術語對照表。
    回傳 {jp_term: {"tw": str, "en": str, "category": str}} 的對照字典。
    若表格為空或連線失敗，拋出 RuntimeError（pipeline 中止）。
    結果在模組層級快取，同一程序只查詢一次。
    """
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE

    url = (os.environ.get("SUPABASE_URL") or "").strip()
    key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY 未設定")

    from supabase import create_client
    client = create_client(url, key)

    result = client.table("knowledge_base").select("jp_term, tw_term, en_term, category").execute()

    kb = {
        row["jp_term"]: {
            "tw": row["tw_term"],
            "en": row["en_term"],
            "category": row["category"],
        }
        for row in result.data
    }

    if not kb:
        raise RuntimeError(
            "knowledge_base 表格中沒有有效的術語資料。"
            "請先執行 python scripts/migrate_kb.py 或在 Supabase 中加入術語後再執行 FFXIV 分析。"
        )

    logger.info("知識庫載入完成：%d 個術語", len(kb))
    _KB_CACHE = kb
    return kb


def _check_kb_misses(text: str, kb: dict) -> None:
    """
    掃描 Gemini 回應中未被知識庫收錄的術語：
    1. [[term]] 標記 — 模型遵守規則時自行標記的未知術語
    2. 裸出現的片假名序列（3 字以上）— 漏標的備援掃描
    兩類皆累積至 _KB_MISS_ACCUMULATOR 供執行結束時提示使用者。
    """
    # 優先：模型用 [[term]] 明確標記的未知術語
    for match in _KB_HIGHLIGHT.findall(text):
        term = match.strip()
        if term and term not in kb:
            logger.warning("[KB MISS] 未知詞彙：%s", term)
            _KB_MISS_ACCUMULATOR.add(term)

    # 備援：未加標記但出現在回應中的片假名序列
    for match in set(_KB_KATAKANA.findall(text)):
        if match not in kb and match not in _KB_MISS_ACCUMULATOR:
            logger.warning("[KB MISS] 未知詞彙（未標記）：%s", match)
            _KB_MISS_ACCUMULATOR.add(match)


def get_kb_miss_summary() -> list:
    """回傳本次執行中所有 [KB MISS] 術語列表，供主程式在執行結束時提示使用者。"""
    return sorted(_KB_MISS_ACCUMULATOR)


def _get_api_key():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY 環境變數未設定")
    return key


# ── Embedding Dedup (traffic pipeline) ───────────────────────────────────────

def generate_embedding(text: str) -> list | None:
    """
    Call Gemini Embedding 2 to produce a 768-dim vector for the given text.
    Returns None on any API failure so callers can skip similarity checks gracefully.
    """
    api_key = _get_api_key()
    url = GEMINI_EMBED_URL.format(api_key=api_key)
    payload = {
        "content": {"parts": [{"text": f"task: semantic_similarity\n\n{text[:4000]}"}]},
        "outputDimensionality": EMBED_DIMENSIONS,
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
    except Exception as e:
        logger.warning("[embedding] 生成失敗：%s", e)
        return None


def _parse_embedding(val) -> list | None:
    """Normalize a pgvector string '[x,y,...]' or list to a list of floats."""
    if val is None:
        return None
    if isinstance(val, list):
        return [float(x) for x in val]
    if isinstance(val, str):
        try:
            return [float(x) for x in val.strip("[] \t\n").split(",")]
        except Exception:
            return None
    return None


def _cosine_similarity(a: list, b: list) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def attach_embeddings(candidates: list) -> None:
    """**這是評分路徑上唯一不純的一步**：對缺向量的候選呼叫 Gemini（會產生成本與網路相依）。

    從 `embed_dedup` 抽出來，好讓後者是純函數、能在沒有憑證的情況下離線測
    （BACKLOG #4：純運算與外部呼叫不該共處，憲章 V）。就地改寫 `candidates`，
    因為 `publish()` 之後要從同一批 dict 讀 `embedding` 存進 DB。
    """
    for a in candidates:
        if "embedding" not in a:
            text = (a.get("title") or "") + "\n" + (a.get("summary") or "")[:300]
            a["embedding"] = generate_embedding(text)
        else:
            a["embedding"] = _parse_embedding(a["embedding"])


def embed_dedup(candidates: list, buffer_articles: list, threshold: float = 0.88) -> list:
    """
    Deduplicate candidates by cosine similarity of Gemini embeddings.
    Checks each candidate against: (1) buffer articles already in the weekly DB,
    (2) other candidates in the same batch.
    Falls back to returning all candidates if embeddings are unavailable.

    **純函數**——不打任何外部服務。向量要先由 `attach_embeddings()` 補上；
    呼叫端漏掉那一步時本函式會**明確警告**（見下），不會靜靜地不去重。
    """
    if not candidates:
        return candidates

    # 只做解析（純），不生成。⚠️ 「沒有 embedding 鍵」與「生成失敗（值為 None）」
    # 必須分開報：前者代表呼叫端漏了 attach_embeddings，後者是 API 掛掉。
    # 兩者的後果相同（不去重）但成因不同，混在一起會讓接線錯誤看起來像 API 故障。
    never_attempted = sum(1 for a in candidates if "embedding" not in a)
    if never_attempted:
        logger.warning(
            "[embed_dedup] %d/%d 篇候選沒有 embedding 欄位——呼叫端可能漏了 "
            "attach_embeddings()，這批不會被去重",
            never_attempted, len(candidates),
        )
    for a in candidates:
        a["embedding"] = _parse_embedding(a.get("embedding"))

    has_embed = [a for a in candidates if a.get("embedding") is not None]
    no_embed = [a for a in candidates if a.get("embedding") is None]

    # 缺鍵的那批解析後也是 None，扣掉才不會把接線錯誤重複算成 API 故障。
    generation_failed = len(no_embed) - never_attempted
    if generation_failed:
        logger.warning(
            "[embed_dedup] %d/%d 篇候選的向量生成失敗，這批不會被去重"
            "（逐筆原因見上方 [embedding] 生成失敗）",
            generation_failed, len(candidates),
        )

    if not has_embed:
        logger.warning("[embed_dedup] 所有候選都沒有可用向量，保留所有")
        return candidates

    for a in buffer_articles:
        if a.get("embedding") is not None:
            a["embedding"] = _parse_embedding(a["embedding"])
    buffer_with_embed = [a for a in buffer_articles if a.get("embedding") is not None]

    # Step 1: drop candidates already covered by the week's buffer
    after_buffer: list = []
    for a in has_embed:
        matched = next(
            (b for b in buffer_with_embed
             if _cosine_similarity(a["embedding"], b["embedding"]) >= threshold),
            None,
        )
        if matched:
            logger.info(
                "[embed_dedup] 已在緩衝區，略過（相似度≥%.2f）：%s",
                threshold, a.get("title", ""),
            )
        else:
            after_buffer.append(a)

    # Step 2: pairwise dedup within the batch; keep the article with more summary content
    deduped: list = []
    for a in after_buffer:
        dup_idx = next(
            (j for j, kept in enumerate(deduped)
             if _cosine_similarity(a["embedding"], kept["embedding"]) >= threshold),
            None,
        )
        if dup_idx is None:
            deduped.append(a)
        else:
            kept_len = len(deduped[dup_idx].get("summary") or "")
            cand_len = len(a.get("summary") or "")
            if cand_len > kept_len:
                logger.info("[embed_dedup] 批次去重，保留較詳細版本：%s", a.get("title", ""))
                deduped[dup_idx] = a
            else:
                logger.info("[embed_dedup] 批次去重，略過：%s", a.get("title", ""))

    result = deduped + no_embed
    skipped = len(candidates) - len(result)
    logger.info("[embed_dedup] 結果：%d → %d 篇（略過 %d）", len(candidates), len(result), skipped)
    return result


def _retry_after_seconds(resp, attempt, status_code):
    """
    503/UNAVAILABLE 常持續數小時至數日（Google 端容量問題），需較長指數退避。
    參考：https://ai.google.dev/gemini-api/docs/troubleshooting
    """
    if resp is not None:
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return min(max(int(ra), 1), 120)
            except ValueError:
                pass
    if status_code == 429:
        return min(10 * (2 ** (attempt - 1)), 120)
    if status_code in (500, 502, 503, 504):
        # 第 1 次約 10s，之後 20、40、80… 上限 120s，略加抖動避免同步重試
        base = min(10 * (2 ** (attempt - 1)), 120)
        return base + random.uniform(0, 4)
    return min(5 * attempt, 30)


def _call_gemini(prompt, api_key, retries=None, system_prompt=None):
    if retries is None:
        retries = int(os.environ.get("GEMINI_MAX_RETRIES", "8"))
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    url = GEMINI_API_URL.format(model=GEMINI_MODEL, api_key=api_key)
    try:
        maxOutputTokens = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
    except ValueError:
        maxOutputTokens = 8192
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": maxOutputTokens,
        },
    }

    last_status = None
    for attempt in range(1, retries + 1):
        resp = None
        try:
            resp = requests.post(url, json=payload, timeout=120)
            last_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                candidate = data["candidates"][0]
                finish_reason = candidate.get("finishReason", "")
                if finish_reason == "MAX_TOKENS":
                    logger.warning(
                        "⚠️  輸出被截斷（MAX_TOKENS），已用 maxOutputTokens=%s；可再提高環境變數 GEMINI_MAX_OUTPUT_TOKENS 或降低 max_articles",
                        maxOutputTokens,
                    )
                text = candidate["content"]["parts"][0]["text"]
                return text
            wait = _retry_after_seconds(resp, attempt, resp.status_code)
            logger.warning(
                "Gemini API 回應 %s（第 %d/%d 次），%.0f 秒後重試：%s",
                resp.status_code,
                attempt,
                retries,
                wait,
                resp.text[:200].replace("\n", " "),
            )
            time.sleep(wait)
        except requests.RequestException as e:
            logger.warning("請求失敗（第 %d/%d 次）：%s", attempt, retries, e)
            wait = _retry_after_seconds(resp, attempt, 0)
            time.sleep(wait)

    if last_status is not None:
        logger.error("Gemini 在 %d 次重試後仍失敗（最後 HTTP %s）", retries, last_status)
    return None


def _parse_response(text):
    """
    解析 Gemini 回傳的固定格式文字。
    支援單行格式（正常）和多行 fallback（Gemini 偶爾換行時）。
    """
    result = {
        "summary": "",
        "analysis": "",
        "importance": "中",
        "importance_reason": "",
        "tags": [],
        "location": "",
    }

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("摘要："):
            result["summary"] = line[3:].strip()
        elif line.startswith("分析："):
            result["analysis"] = line[3:].strip()
        elif line.startswith("重要性："):
            val = line[4:].strip()
            if val and val[0] in ("高", "中", "低"):
                result["importance"] = val[0]
        elif line.startswith("重要性原因："):
            result["importance_reason"] = line[6:].strip()
        elif line.startswith("標籤："):
            raw_tags = line[3:].strip()
            result["tags"] = [t.strip() for t in raw_tags.replace("、", ",").replace("，", ",").split(",") if t.strip()]
        elif line.startswith("地區："):
            result["location"] = line[3:].strip()

    # fallback：分析欄位多行合併
    if not result["analysis"]:
        lines = text.splitlines()
        collecting = False
        buf = []
        known_tags = ("摘要：", "重要性：", "重要性原因：", "地區：")
        for line in lines:
            line = line.strip()
            if line.startswith("分析："):
                collecting = True
                remainder = line[3:].strip()
                if remainder:
                    buf.append(remainder)
            elif collecting:
                if any(line.startswith(t) for t in known_tags) or not line:
                    break
                buf.append(line)
        if buf:
            result["analysis"] = "".join(buf)

    # fallback：摘要欄位多行合併
    if not result["summary"]:
        lines = text.splitlines()
        collecting = False
        buf = []
        known_tags = ("分析：", "重要性：", "重要性原因：", "地區：")
        for line in lines:
            line = line.strip()
            if line.startswith("摘要："):
                collecting = True
                remainder = line[3:].strip()
                if remainder:
                    buf.append(remainder)
            elif collecting:
                if any(line.startswith(t) for t in known_tags) or not line:
                    break
                buf.append(line)
        if buf:
            result["summary"] = "".join(buf)

    if not result["summary"] and text:
        result["summary"] = text[:300]

    return result


def analyze_article(article, api_key):
    content_type = article.get("content_type", "traffic")
    user_tags = article.get("user_tags", [])

    if article.get("source_type") == "youtube":
        all_tags = sorted(set(DEFAULT_TAGS) | set(user_tags))
        prompt = YOUTUBE_ANALYSIS_PROMPT_TEMPLATE.format(
            title=article.get("title", ""),
            source=article.get("source", ""),
            summary=article.get("summary", "（無逐字稿）"),
            link=article.get("link", ""),
            available_tags="、".join(all_tags),
        )
        raw = _call_gemini(prompt, api_key, system_prompt=YOUTUBE_SYSTEM_PROMPT)
    elif content_type == "ffxiv":
        kb = load_knowledge_base()
        kb_lines = [
            f"| {jp} | {v['tw']} | {v['en']} | {v['category']} |"
            for jp, v in kb.items()
        ]
        kb_block = "| JP | 繁中 | EN | 類別 |\n|----|----|----|----||\n" + "\n".join(kb_lines)
        all_tags = sorted(set(FFXIV_DEFAULT_TAGS) | set(user_tags))
        prompt = FFXIV_ANALYSIS_TEMPLATE.format(
            title=article.get("title", ""),
            source=article.get("source", ""),
            summary=article.get("summary", "（無摘要）"),
            link=article.get("link", ""),
            knowledge_base=kb_block,
            available_tags="、".join(all_tags),
        )
        raw = _call_gemini(prompt, api_key, system_prompt=FFXIV_SYSTEM_PROMPT)
        if raw:
            _check_kb_misses(raw, kb)
    else:
        all_tags = sorted(set(DEFAULT_TAGS) | set(user_tags))
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            title=article.get("title", ""),
            source=article.get("source", ""),
            summary=article.get("summary", "（無摘要）"),
            link=article.get("link", ""),
            available_tags="、".join(all_tags),
        )
        raw = _call_gemini(prompt, api_key)

    if raw:
        article["analysis"] = _parse_response(raw)
        logger.info("✓ 分析完成：%s...", article["title"][:30])
    else:
        logger.warning("✗ 分析失敗：%s", article["title"][:30])
        article["analysis"] = {
            "summary": "（分析失敗）",
            "analysis": "（分析失敗）",
            "importance": "中",
            "importance_reason": "（無法取得分析）",
            "tags": [],
        }

    return article


def analyze_all(articles, delay=6):
    """
    依序分析所有文章。
    delay: 每次請求間隔秒數。
      gemini-2.5-flash（預設）:  10 RPM → delay=6s
      gemini-2.5-flash-lite:     15 RPM → delay=4s
      gemini-2.5-pro:             5 RPM → delay=12s
    """
    api_key = _get_api_key()
    total = len(articles)
    eta_min = round(total * delay / 60, 1)
    try:
        mot_log = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "8192"))
    except ValueError:
        mot_log = 8192
    # 完整模型 id 若在 GitHub Secrets 與日誌相同字串會被遮罩；尾段通常足供除錯且不易觸發遮罩
    model_hint = GEMINI_MODEL.rpartition("-")[-1] if GEMINI_MODEL else "default"
    logger.info(
        "=== 開始 AI 分析，模型尾段=%s，maxOutputTokens=%s，共 %d 篇（間隔 %ds，預估 %.1f 分鐘）===",
        model_hint,
        mot_log,
        total,
        delay,
        eta_min,
    )

    results = []
    for i, article in enumerate(articles):
        remaining = total - i - 1
        logger.info("[%d/%d] 分析中（完成後還剩 %d 篇，約 %.0f 秒）：%s",
                    i + 1, total, remaining,
                    remaining * delay,
                    article["title"][:40])
        result = analyze_article(article, api_key)
        results.append(result)
        if i < total - 1:
            time.sleep(delay)

    order = {"高": 0, "中": 1, "低": 2}
    results.sort(key=lambda x: order.get(x.get("analysis", {}).get("importance", "中"), 1))

    logger.info("=== AI 分析完成 ===")
    return results


HOT_TOPIC_SYSTEM_PROMPT = (
    "你是台灣交通安全結構性分析師，專精機車與道路安全議題。"
    "本分析採「結構性懷疑論」方法：對官方新聞稿的歸因假設持系統性質疑，"
    "但對未經文本支撐的政經推論保持否決權。"
    "所有結論必須標註文本依據（Article ID）。"
    "請用繁體中文回應，格式務必嚴格按照指示，不要加入任何額外說明。"
)

HOT_TOPIC_PROMPT_TEMPLATE = """以下是本週「{topic_label}」類別的熱點新聞群（共 {article_count} 篇），請進行三軸結構性分析。
所有引證以 Article ID 標注（如 [1]、[2]）。

{article_list}

【分析指令】

▌ A：雙軸因果矩陣
將本批文章依以下三級分類，並挑選最具代表性的 1-2 篇做深度解構：
- [工程主導]：移除工程缺陷後，相同人因錯誤不致死或傷亡大幅降低
- [雙因交織]：工程缺陷放大人因錯誤致死率，兩者缺一不可
- [人因主導]：有直接證據顯示個體行為是充分條件，但必須追問：「此行為為何在台灣路網允許普遍化？」

評估工程缺陷必須對照具體標準（禁止空洞泛稱「設計不良」）：
- 日本生活道路30：物理限速工程（車道縮減、實體減速丘、幾何彎線強迫減速）
- 荷蘭持續路：功能單一化（幹支道邊界清晰、消滅直行與轉向的同相位交織）

引用外國原則時的轉譯紀律（防「掛名詞、漂語感」）：
- 講機制，不掛名詞：須寫出該原則的物理因果（如動能 ½mv² 落差），不可只標原則名稱當權威背書。
- 用原則本身的變數：荷蘭同質性原則是「依速度與質量分隔」，禁止改寫成「依車種分隔／車種混流」——那是台灣「車種分流」框架，語感相反。
- 多解法不可只列一條：同質性的解法有二（實體分隔「或」降速至 30km/h），少寫一條會讓讀者倒推出相反結論，兩條都要交代。
- 撞詞須切割：若用到「分隔／分流」這類與台灣政策口號同形的詞，須註明此處指「依速度的實體保護」，非車種側分。
- 驗收標準：結論不得能被外行讀者讀成「車種分流是對的」。

人因指控若僅為記者推斷或警方口述，標注 [推斷，非直接證據]。

▌ B：條件式政經審查（嚴格觸發制）
僅在文本明確提及以下內容時才啟動，否則填「無觸發條件，不適用」：
- 科技執法、區間測速、罰款機制
- 商圈違停、人行道受阻、路口幾何無法重構

▌ C：統計論述偏誤審查（嚴格觸發制）
僅在文本明確出現「官方政策成效宣稱」或「政府／主管機關統計數據」時才啟動，否則填「無官方統計宣稱，不適用」，不要逐項臆測。
啟動時就以下面向逐項檢查，並只列出「有發現」的偏誤（未發現的項目不要列、不要填資訊不足）：
- 工程問題個體化歸因：把工程缺陷導致的事故歸因為個體行為
- 未交代分母的統計：事故率下降未交代流量分母
- 時間窗口操縱：統計區間迴避異常高峰年份
- 迴避路權移轉數據：未交代路權移轉至周邊路線的可能

【輸出格式】（嚴格照以下格式逐行輸出，每行一個欄位，不可合併或換行）

### 一、 事故因果解構
交織度分布：[工程主導] __篇 ／ [雙因交織] __篇 ／ [人因主導] __篇
代表個案：（填 Article ID）
人因變數：（說明個體決策，標注 [直接證據] 或 [推斷]）
工程變數：（對照日本生活道路30或荷蘭持續路，以速度／質量等原則本身的變數描述差距，禁用車種框架）
交織說明：（說明工程如何放大人因，或說明工程無關）

### 二、 政經變數審查
觸發條件：（填觸發的 Article ID，或「無觸發條件，不適用」）
審查結果：（若適用，分析制度慣性；無觸發則填「不適用」）

### 三、 官方論述偏誤
觸發條件：（填出現官方政策成效宣稱／政府統計數據的 Article ID，或「無官方統計宣稱，不適用」）
審查結果：（觸發時逐項以 □ 列出有發現的偏誤並附 [Article ID]；未觸發填「不適用」，不要列資訊不足）
"""


def analyze_hot_topic(bucket_articles: list, topic_label: str, week_start_date: str) -> tuple:
    """
    Generate a deep-analysis report for a hot topic bucket using Gemini.
    Selects top 10 articles by initial_quality_score for the prompt.
    Returns (report_text, ordered_links) where ordered_links[i] corresponds to [i+1] in the report.
    Returns ("", []) on failure.
    Caller is responsible for respecting rate limits (existing 2.5s delay convention).
    """
    api_key = _get_api_key()

    top_articles = sorted(
        bucket_articles,
        key=lambda a: float(a.get("initial_quality_score") or 0),
        reverse=True,
    )[:10]

    ordered_articles = [
        {
            "title": a.get("title", ""),
            "link": a.get("link", ""),
            "summary": (a.get("summary", "") or "")[:200],
        }
        for a in top_articles
    ]

    lines = []
    for i, a in enumerate(top_articles, 1):
        title = a.get("title", "（無標題）")
        body = (a.get("summary", "") or "")[:600]
        lines.append(f"[{i}] 標題：{title}\n    摘要：{body}")

    article_list = "\n\n".join(lines)
    prompt = HOT_TOPIC_PROMPT_TEMPLATE.format(
        topic_label=topic_label,
        article_count=len(top_articles),
        article_list=article_list,
    )

    raw = _call_gemini(prompt, api_key, system_prompt=HOT_TOPIC_SYSTEM_PROMPT)
    if raw:
        logger.info("✓ 熱點分析完成：%s (%s)", topic_label, week_start_date)
        return raw.strip(), ordered_articles

    logger.warning("✗ 熱點分析失敗：%s (%s)", topic_label, week_start_date)
    return "", []


_DEDUP_SYSTEM_PROMPT = (
    "你是新聞去重複助手。你的任務是找出描述「完全相同事件」的重複報導。\n"
    "重複的判斷標準：同一地點＋同一事件類型＋同一政府機關或當事人出現，即視為同一事件，"
    "即使措辭不同（如「騎士」vs「網紅」、「函警」vs「報警舉發」）。\n"
    "不算重複的情形：主題相似但事件明顯不同（不同起事故、不同地點、不同當事人）。\n"
    "候選間互為重複時，只保留摘要最詳細的一篇。\n"
    "只回覆 JSON，不加任何說明或 markdown 符號。"
)


def dedup_same_event(candidates: list, buffer_articles: list) -> list:
    """
    Use Gemini to drop candidates that describe the same event as each other
    or as articles already in the weekly buffer.
    Jaccard dedup should run first; this handles the grey zone that token overlap misses.
    Falls back to returning all candidates on any error.
    """
    if not candidates:
        return candidates

    import json

    api_key = _get_api_key()

    buffer_lines = []
    for i, a in enumerate(buffer_articles[-30:], 1):
        excerpt = (a.get("summary") or "")[:120].replace("\n", " ")
        buffer_lines.append(f"[B{i}] {a.get('title', '')}\n摘要：{excerpt}")
    buffer_section = "\n\n".join(buffer_lines) or "（本週尚無已收錄文章）"

    candidate_lines = []
    for i, a in enumerate(candidates, 1):
        excerpt = (a.get("summary") or "")[:120].replace("\n", " ")
        candidate_lines.append(f"[N{i}] {a.get('title', '')}\n摘要：{excerpt}")
    candidates_section = "\n\n".join(candidate_lines)

    prompt = (
        f"本週緩衝區（已收錄）：\n\n{buffer_section}\n\n"
        f"今日候選（待決定）：\n\n{candidates_section}\n\n"
        "請找出今日候選中，與緩衝區或其他候選描述「完全相同事件」的重複報導。\n"
        "排除條件：同一地點＋同一事件類型＋同一政府機關或當事人出現，即可排除，措辭不同亦算重複。\n"
        "保留條件：不同起事故（不同當事人、不同地點、不同日期）一律保留。\n"
        "候選間互為重複時，保留摘要最詳細的一篇。\n\n"
        '回覆 JSON，必須包含 exclude（排除的 N 編號）與 reasons（排除原因）：\n'
        '{"exclude": ["N3"], "reasons": {"N3": "與 N1 描述同一天同一地點的事故"}, "keep": ["N1", "N2"]}'
    )

    logger.info(
        "[dedup_same_event] 送出：%d 候選，%d 緩衝區文章",
        len(candidates), len(buffer_articles),
    )
    logger.debug("[dedup_same_event] prompt:\n%s", prompt)

    try:
        raw = _call_gemini(prompt, api_key, system_prompt=_DEDUP_SYSTEM_PROMPT)
        logger.debug("[dedup_same_event] 原始回覆：%s", raw)
        if not raw:
            logger.warning("[dedup_same_event] Gemini 回傳空回覆，保留所有候選")
            return candidates

        # Find the outermost JSON object using brace counting (handles nested objects)
        start = raw.find('{')
        if start == -1:
            logger.warning("[dedup_same_event] 無法解析回覆（找不到 JSON），保留所有候選\n原始回覆：%s", raw)
            return candidates
        depth, end = 0, -1
        for i, ch in enumerate(raw[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end == -1:
            logger.warning("[dedup_same_event] JSON 括號不匹配，保留所有候選\n原始回覆：%s", raw)
            return candidates

        result = json.loads(raw[start:end + 1])
        exclude_ids = {s for s in result.get("exclude", []) if isinstance(s, str)}
        reasons = result.get("reasons", {})
        for nid, reason in reasons.items():
            if nid in exclude_ids:
                logger.info("[dedup_same_event] 排除 %s：%s", nid, reason)

        filtered = [a for i, a in enumerate(candidates, 1) if f"N{i}" not in exclude_ids]
        skipped = len(candidates) - len(filtered)
        logger.info("[dedup_same_event] 結果：%d → %d 篇（略過 %d）", len(candidates), len(filtered), skipped)
        return filtered

    except Exception as e:
        logger.warning("[dedup_same_event] LLM 去重複失敗，保留所有候選：%s", e)
        return candidates


# ── Traffic Pipeline: Clustering + Scoring (US3) ─────────────────────────────

def _article_word_count(article: dict) -> int:
    body = article.get("summary", "") or ""
    return len((article.get("title", "") + body).replace(" ", ""))


def _date_part(published: str) -> str:
    return published[:10] if published else ""


def cluster_traffic_articles(articles: list, config: dict) -> dict:
    """
    Group traffic articles into topic buckets by Jaccard similarity within each
    major_category.

    - Jaccard > merge_threshold: near-duplicate pair — keep higher word-count article (FR-013)
    - Jaccard in [cluster_lower, merge_threshold]: same topic bucket (FR-014)
    - Otherwise: new independent bucket

    Returns {bucket_id: [article_dicts]}.
    """
    from src.filter import normalise_title, compute_jaccard

    merge_threshold = config.get("jaccard", {}).get("merge_threshold", 0.45)
    cluster_lower = config.get("jaccard", {}).get("cluster_lower", 0.20)

    # Group by major_category
    by_category: dict = {}
    for article in articles:
        cat = article.get("major_category", "uncategorised")
        by_category.setdefault(cat, []).append(article)

    all_buckets: dict = {}
    bucket_counter = 0

    for cat_articles in by_category.values():
        token_sets = {id(a): normalise_title(a.get("title", "")) for a in cat_articles}

        # 1. Deduplicate: sort by word count desc; retain article only if no
        #    already-retained article has Jaccard > merge_threshold with it.
        sorted_articles = sorted(cat_articles, key=_article_word_count, reverse=True)
        deduplicated: list = []
        for article in sorted_articles:
            ts_a = token_sets[id(article)]
            is_dup = any(
                compute_jaccard(ts_a, token_sets[id(r)]) > merge_threshold
                for r in deduplicated
            )
            if not is_dup:
                deduplicated.append(article)

        # 2. Cluster: assign to an existing bucket if any member scores in
        #    [cluster_lower, merge_threshold]; start a new bucket otherwise.
        cat_buckets: dict = {}
        for article in deduplicated:
            ts_a = token_sets[id(article)]
            assigned = False
            for bid, bucket_articles in cat_buckets.items():
                for b_article in bucket_articles:
                    score = compute_jaccard(ts_a, token_sets[id(b_article)])
                    if cluster_lower <= score <= merge_threshold:
                        bucket_articles.append(article)
                        assigned = True
                        break
                if assigned:
                    break
            if not assigned:
                bid = f"bucket_{bucket_counter}"
                bucket_counter += 1
                cat_buckets[bid] = [article]

        all_buckets.update(cat_buckets)

    return all_buckets


def score_topic_buckets(buckets: dict, config: dict) -> dict:
    """
    Compute the cumulative momentum score for each bucket:
    score = Σ(quality_scores) × log(distinct_sources + 1) × log(distinct_days + 1)

    Returns {bucket_id: score}.
    """
    import math

    scores: dict = {}
    for bid, bucket_articles in buckets.items():
        quality_sum = sum(
            float(a.get("initial_quality_score") or 0) for a in bucket_articles
        )
        distinct_sources = len({a.get("source", "") for a in bucket_articles})
        distinct_days = len({
            _date_part(a.get("published", ""))
            for a in bucket_articles
            if a.get("published")
        })
        score = (
            quality_sum
            * math.log(distinct_sources + 1)
            * math.log(max(distinct_days, 1) + 1)
        )
        scores[bid] = score

    return scores


def select_hot_topics(bucket_scores: dict, config: dict) -> list:
    """
    Return at most max_hot_topics bucket_ids with score >= min_threshold,
    sorted by score descending.
    """
    topic_cfg = config.get("topic_scoring", {})
    min_threshold = float(topic_cfg.get("min_threshold", 1.5))
    max_hot_topics = int(topic_cfg.get("max_hot_topics", 3))

    qualified = [
        (bid, score) for bid, score in bucket_scores.items()
        if score >= min_threshold
    ]
    qualified.sort(key=lambda x: x[1], reverse=True)
    return [bid for bid, _ in qualified[:max_hot_topics]]


# ── Novelty gate (feature 009) ────────────────────────────────────────────────

_TOPIC_LABEL_SEP = " · "


def topic_token_signature(bucket_articles: list, top_k: int = 8) -> list:
    """Representative tokens for a bucket: the most frequent normalise_title tokens
    across its articles (up to top_k). Used for cross-week topic identity matching."""
    from collections import Counter
    from src.filter import normalise_title

    counter: Counter = Counter()
    for a in bucket_articles:
        counter.update(normalise_title(a.get("title", "")))
    return [tok for tok, _ in counter.most_common(top_k)]


def _bucket_latest_date(bucket_articles: list) -> str:
    """Latest published date (YYYY-MM-DD) among bucket articles; '' if none."""
    dates = [_date_part(a.get("published", "")) for a in bucket_articles if a.get("published")]
    return max(dates) if dates else ""


def _topic_label_category(topic_label: str) -> str:
    """Extract the major_category prefix from a topic_label ("<category> · <term>").
    Pre-009 reports stored the bare category, so a label without the separator is
    returned as-is."""
    return topic_label.split(_TOPIC_LABEL_SEP, 1)[0] if _TOPIC_LABEL_SEP in topic_label else topic_label


def _match_prior_basis(major_category: str, signature: list, prior_reports: list,
                       similarity_threshold: float) -> dict | None:
    """Most recent prior report of the same major_category whose
    topic_token_signature has Jaccard ≥ threshold with `signature`.
    prior_reports are assumed ordered week_start_date DESC. Returns basis or None."""
    from src.filter import compute_jaccard

    sig_set = frozenset(signature)
    for r in prior_reports:
        if _topic_label_category(r.get("topic_label", "")) != major_category:
            continue
        prior_sig = frozenset(r.get("topic_token_signature") or [])
        if compute_jaccard(sig_set, prior_sig) >= similarity_threshold:
            return r
    return None


def passes_novelty(bucket_score: float, bucket_latest_date: str,
                   prior_basis: dict | None, config: dict) -> bool:
    """No prior basis → novel (first-time, FR-003). Otherwise require BOTH:
       (a) bucket_score ≥ prior.cumulative_score × (1 + novelty_growth_pct), and
       (b) bucket_latest_date strictly later than prior.latest_source_date
           (a new publication day since the last report)."""
    if prior_basis is None:
        return True
    p = float(config.get("topic_scoring", {}).get("novelty_growth_pct", 0.5))
    last_score = float(prior_basis.get("cumulative_score") or 0)
    if bucket_score < last_score * (1 + p):
        return False
    last_date = prior_basis.get("latest_source_date") or ""
    return bool(bucket_latest_date) and bucket_latest_date > last_date


def select_hot_topics_with_novelty(buckets: dict, bucket_scores: dict,
                                   prior_reports: list, config: dict) -> list:
    """Gate-then-cap (feature 009): gate every bucket scoring ≥ min_threshold through
    the novelty check (matched against same-category prior reports by signature
    similarity), then take the top max_hot_topics survivors by score.
    Returns bucket_ids. prior_reports=[] → behaves like first-time (all novel)."""
    topic_cfg = config.get("topic_scoring", {})
    default_min = float(topic_cfg.get("min_threshold", 1.5))
    category_min = topic_cfg.get("category_min_threshold") or {}
    max_hot_topics = int(topic_cfg.get("max_hot_topics", 3))
    sim_threshold = float(config.get("topic_identity", {}).get("similarity_threshold", 0.3))

    survivors: list = []
    for bid, score in bucket_scores.items():
        articles = buckets.get(bid, [])
        major_category = articles[0].get("major_category", bid) if articles else bid
        # Low-cadence categories (e.g. 道安政策) can carry a lower per-category
        # threshold so their small buckets aren't blocked by the global bar tuned
        # for high-volume categories; falls back to the global min_threshold.
        threshold = float(category_min.get(major_category, default_min))
        if score < threshold:
            continue
        signature = topic_token_signature(articles)
        latest_date = _bucket_latest_date(articles)
        prior = _match_prior_basis(major_category, signature, prior_reports, sim_threshold)
        is_novel = passes_novelty(score, latest_date, prior, config)
        logger.info(
            "  novelty[%s] cat=%s score=%.3f thr=%.2f prior=%s newer_than=%s → %s",
            bid, major_category, score, threshold,
            "yes" if prior else "none",
            (prior.get("latest_source_date") if prior else "-"),
            "PASS" if is_novel else "suppress",
        )
        if is_novel:
            survivors.append((bid, score))

    survivors.sort(key=lambda x: x[1], reverse=True)
    return [bid for bid, _ in survivors[:max_hot_topics]]


# ── Category digest (feature 010) ─────────────────────────────────────────────

DIGEST_SYSTEM_PROMPT = (
    "你是台灣交通政策動態彙整編輯，專精道路安全政策、法規與治理議題。"
    "彙整原則：各事件互相獨立，禁止在無文本依據下建立事件間的因果、趨勢或風向連結；"
    "所有敘述必須標註文本依據（Article ID，如 [1]）。"
    "請用繁體中文回應，格式務必嚴格按照指示，不要加入任何額外說明。"
)

DIGEST_PROMPT_TEMPLATE = """以下是「{topic_label}」在 {date_start} 至 {date_end} 期間累積的 {article_count} 篇報導。
這些是多個互相獨立的事件，請產出一份動態總覽（digest），不是單一事件深挖。

{article_list}

【彙整指令】
- 依主題相近度將文章分組（例：地方道安會報動態／中央政策與法規／民間倡議與研究），每組一節。
- 每節逐事件一短段：時間、行為者、決策或訴求內容，段末標注 [Article ID]。
- 多篇報導同一事件時合併為一段，並列出全部 Article ID。
- 不同事件之間沒有文本依據時，禁止寫成趨勢、因果或「顯示…風向」的連結語。

【輸出格式】（嚴格照以下格式輸出）

### 本期概覽
（2-3 句：本期共幾個獨立事件、涵蓋哪些面向。不做趨勢推論。）

### 分節彙整
（依分組逐節逐事件輸出，每段結尾標 [Article ID]）

### 待追蹤
（列出文本中明示「後續將…」「預計…」的未完成事項，各附 [Article ID]；無則填「本期無」）
"""


def select_digest_pool(articles: list, category: str, digest_cfg: dict,
                       excluded_links: set) -> tuple:
    """
    組出某低頻類別的 digest 池（純函數、零 I/O）。

    Returns (selected, pool_all, effective_count):
    - pool_all: 該類別全部未排除文章（含低於品質下限者）——消耗（清池）用。
    - selected: quality ≥ quality_floor 者依 quality 降冪取前 max_articles 篇——選材。
    - effective_count: quality ≥ quality_floor 的篇數——觸發判斷用（呼叫端比對 trigger_count）。

    池的成員判定吃**一組**類別（feature 013）：`{category} ∪ include_categories`。
    用集合而非 list 串接，故清單含主類別自己時自動去重、順序不影響輸出。
    缺 `include_categories` ⇒ 空清單 ⇒ 集合退化為 `{category}` ⇒ 行為與本功能
    不存在時逐篇相同。**不改動輸入物件**：匯流只影響成員判定，不寫回
    `major_category`，buffer list 的細分類因此保留。
    """
    quality_floor = float(digest_cfg.get("quality_floor", 0.18))
    max_articles = int(digest_cfg.get("max_articles", 15))
    categories = {category} | set(digest_cfg.get("include_categories") or [])

    pool_all = [
        a for a in articles
        if a.get("major_category") in categories
        and (a.get("link") or "") not in excluded_links
    ]
    effective = [
        a for a in pool_all
        if float(a.get("initial_quality_score") or 0) >= quality_floor
    ]
    effective.sort(key=lambda a: float(a.get("initial_quality_score") or 0), reverse=True)
    return effective[:max_articles], pool_all, len(effective)


def log_digest_pool_composition(logger, category: str, digest_cfg: dict,
                                pool_all: list) -> None:
    """印出 digest 池的逐類別組成（013 C3），讓「匯流有沒有效」不必查 DB 就能從 log 判讀。

    **零篇的類別也要印**——沉默與「跑了但沒事可做」無法區分，而缺口能躺數週
    正是因為沉默不留腳印。這條規則的誘惑點就在下面那個 join：任何「跳過 0」的
    寫法（`if n`、`most_common()`、過濾 falsy）都會讓零篇類別整行消失，
    所以 `test_composition_prints_zero_count_category` 守的是這裡。

    `logger` 由呼叫端傳入，讓這行仍掛在週跑腳本的 logger 名字下（log 輸出逐字不變）。
    無 `include_categories` ⇒ 不印，與匯流不存在時行為相同。
    """
    from collections import Counter

    merged_cats = list(digest_cfg.get("include_categories") or [])
    if not merged_cats:
        return
    per_cat = Counter(a.get("major_category") for a in pool_all)
    parts = " ＋ ".join(f"{c} {per_cat.get(c, 0)}" for c in [category] + merged_cats)
    logger.info("digest[%s] 池組成：%s = %d", category, parts, len(pool_all))


def analyze_category_digest(pool_articles: list, topic_label: str,
                            week_start_date: str, max_articles: int = 15) -> tuple:
    """
    Generate a multi-event digest report for a low-cadence category using Gemini.
    Same return contract as analyze_hot_topic: (report_text, ordered_links);
    ("", []) on failure. Caller is responsible for the 2.5s inter-request delay.
    """
    api_key = _get_api_key()

    top_articles = sorted(
        pool_articles,
        key=lambda a: float(a.get("initial_quality_score") or 0),
        reverse=True,
    )[:max_articles]
    if not top_articles:
        return "", []

    ordered_articles = [
        {
            "title": a.get("title", ""),
            "link": a.get("link", ""),
            "summary": (a.get("summary", "") or "")[:200],
        }
        for a in top_articles
    ]

    dates = sorted({(a.get("published") or "")[:10] for a in top_articles if a.get("published")})
    date_start, date_end = (dates[0], dates[-1]) if dates else (week_start_date, week_start_date)

    lines = []
    for i, a in enumerate(top_articles, 1):
        title = a.get("title", "（無標題）")
        body = (a.get("summary", "") or "")[:400]
        lines.append(f"[{i}] 日期：{(a.get('published') or '')[:10]}\n    標題：{title}\n    摘要：{body}")

    prompt = DIGEST_PROMPT_TEMPLATE.format(
        topic_label=topic_label,
        article_count=len(top_articles),
        date_start=date_start,
        date_end=date_end,
        article_list="\n\n".join(lines),
    )

    raw = _call_gemini(prompt, api_key, system_prompt=DIGEST_SYSTEM_PROMPT)
    if raw:
        logger.info("✓ 彙整分析完成：%s (%s)", topic_label, week_start_date)
        return raw.strip(), ordered_articles

    logger.warning("✗ 彙整分析失敗：%s (%s)", topic_label, week_start_date)
    return "", []