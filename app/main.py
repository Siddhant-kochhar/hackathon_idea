from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, Optional, List
import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from fastapi import Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

ALPHA_VANTAGE_MCP_URL_ENV = "ALPHA_VANTAGE_MCP_URL"


class QueryRequest(BaseModel):
    message: str


class MCPClient:
    """Minimal JSON-RPC over HTTP client for Alpha Vantage MCP server.

    The official MCP server supports HTTP JSON-RPC at the connection URL. We call tools
    by POSTing a JSON-RPC request with method 'tools.call' and tool-specific params.
    """

    def __init__(self, base_url: str, client: Optional[httpx.AsyncClient] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=30)
        self._id_counter = 0

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    async def call_tool(self, tool: str, params: Dict[str, Any]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool, "arguments": params},
        }
        resp = await self.client.post(self.base_url, json=payload)
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        if "error" in data:
            raise HTTPException(status_code=502, detail=data["error"])
        return data.get("result")


class GeminiClient:
    """Google Gemini AI client for financial news and educational content."""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    async def get_financial_news_summary(self, query: str) -> str:
        """Get AI-powered financial news and insights"""
        prompt = f"""
        You are a financial news analyst AI. The user asked: "{query}"
        
        Provide a helpful response about financial news, market trends, or financial education.
        
        Guidelines:
        - If it's about specific news, provide general market insights and trends
        - If it's about financial concepts, provide educational explanations
        - Keep responses concise but informative (2-3 paragraphs max)
        - Use markdown formatting for better readability
        - Include relevant financial context
        - Don't provide specific investment advice, but educational information is fine
        
        Response:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm having trouble accessing the latest information right now. Error: {str(e)}"
    
    async def get_financial_education(self, query: str) -> str:
        """Get educational content about any topic, with focus on financial/business context when relevant"""
        prompt = f"""
        You are a knowledgeable AI assistant with expertise in finance, business, and general topics. The user asked: "{query}"
        
        Provide a helpful and informative response. If the query is about:
        - **Companies**: Provide business overview, recent developments, market position
        - **Financial topics**: Explain concepts clearly with examples
        - **General topics**: Provide accurate, helpful information
        - **News/current events**: Share relevant insights and context
        
        Guidelines:
        - Be informative and accurate
        - Use markdown formatting for better readability
        - Include relevant details and context
        - Structure content with headers and bullet points when helpful
        - If it's about a specific company, include business model, recent performance, market position
        - Keep responses comprehensive but concise (2-4 paragraphs)
        
        Response:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"I apologize, but I'm having trouble generating educational content right now. Error: {str(e)}"


def pick_tool_for_message(text: str) -> tuple[str, Dict[str, Any]]:
    t = text.lower().strip()

    # Historical performance patterns - past week, month, year performance
    performance_match = re.search(r"(?:past|last)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+performance\s+(?:of\s+)?([A-Za-z\.\-]{1,10})", t)
    if performance_match:
        period_num = int(performance_match.group(1))
        period_type = performance_match.group(2)
        symbol = performance_match.group(3).upper()
        
        # Map to appropriate time series based on period
        if ("day" in period_type and period_num <= 7) or ("week" in period_type and period_num == 1):
            return "TIME_SERIES_INTRADAY", {"symbol": symbol, "interval": "60min", "outputsize": "compact"}
        elif "week" in period_type or ("day" in period_type and period_num <= 30):
            return "TIME_SERIES_DAILY", {"symbol": symbol, "outputsize": "compact"}
        elif "month" in period_type or ("day" in period_type and period_num <= 365):
            return "TIME_SERIES_WEEKLY", {"symbol": symbol}
        else:
            return "TIME_SERIES_MONTHLY", {"symbol": symbol}

    # Intraday data: intraday AAPL 5min
    intraday_match = re.search(r"intraday\s+([A-Za-z\.\-]{1,10})(?:\s+(1min|5min|15min|30min|60min))?", t)
    if intraday_match:
        symbol = intraday_match.group(1).upper()
        interval = intraday_match.group(2) or "5min"
        return "TIME_SERIES_INTRADAY", {"symbol": symbol, "interval": interval, "outputsize": "compact"}

    # Daily data: daily AAPL or daily adjusted AAPL
    daily_match = re.search(r"daily\s+(?:adjusted\s+)?([A-Za-z\.\-]{1,10})", t)
    if daily_match:
        symbol = daily_match.group(1).upper()
        if "adjusted" in t:
            return "TIME_SERIES_DAILY_ADJUSTED", {"symbol": symbol, "outputsize": "compact"}
        return "TIME_SERIES_DAILY", {"symbol": symbol, "outputsize": "compact"}

    # Weekly data: weekly AAPL or weekly adjusted AAPL
    weekly_match = re.search(r"weekly\s+(?:adjusted\s+)?([A-Za-z\.\-]{1,10})", t)
    if weekly_match:
        symbol = weekly_match.group(1).upper()
        if "adjusted" in t:
            return "TIME_SERIES_WEEKLY_ADJUSTED", {"symbol": symbol}
        return "TIME_SERIES_WEEKLY", {"symbol": symbol}

    # Monthly data: monthly AAPL or monthly adjusted AAPL
    monthly_match = re.search(r"monthly\s+(?:adjusted\s+)?([A-Za-z\.\-]{1,10})", t)
    if monthly_match:
        symbol = monthly_match.group(1).upper()
        if "adjusted" in t:
            return "TIME_SERIES_MONTHLY_ADJUSTED", {"symbol": symbol}
        return "TIME_SERIES_MONTHLY", {"symbol": symbol}

    # Bulk quotes: quotes AAPL,MSFT,GOOGL
    bulk_quotes_match = re.search(r"quotes?\s+([A-Za-z\.\-,\s]+)", t)
    if bulk_quotes_match and "," in bulk_quotes_match.group(1):
        symbols = [s.strip().upper() for s in bulk_quotes_match.group(1).split(",")]
        return "REALTIME_BULK_QUOTES", {"symbols": ",".join(symbols[:100])}  # Limit to 100 symbols

    # Price history patterns: "AAPL price history" or "price history AAPL"
    price_history_match = re.search(r"(?:([A-Za-z\.\-]{1,10})\s+)?price\s+history(?:\s+([A-Za-z\.\-]{1,10}))?", t)
    if price_history_match:
        symbol = (price_history_match.group(1) or price_history_match.group(2) or "").upper()
        if symbol:
            return "TIME_SERIES_DAILY", {"symbol": symbol, "outputsize": "compact"}

    # Chart patterns: "chart AAPL" or "AAPL chart"
    chart_match = re.search(r"(?:chart\s+([A-Za-z\.\-]{1,10})|([A-Za-z\.\-]{1,10})\s+chart)", t)
    if chart_match:
        symbol = (chart_match.group(1) or chart_match.group(2)).upper()
        return "TIME_SERIES_DAILY", {"symbol": symbol, "outputsize": "compact"}

    # Top gainers/losers/most active - with specific intent detection
    if "gainers" in t and not ("losers" in t or "most active" in t):
        return "TOP_GAINERS_LOSERS", {"_filter": "gainers"}
    elif "losers" in t and not ("gainers" in t or "most active" in t):
        return "TOP_GAINERS_LOSERS", {"_filter": "losers"}
    elif "most active" in t and not ("gainers" in t or "losers" in t):
        return "TOP_GAINERS_LOSERS", {"_filter": "active"}
    elif (
        "top" in t
        or "top performers" in t
        or "top performance" in t
        or "growing" in t
        or ("gainers" in t and "losers" in t)
        or ("market" in t and ("overview" in t or "summary" in t))
    ):
        return "TOP_GAINERS_LOSERS", {"_filter": "all"}

    # Quote for a ticker like: quote AAPL
    m = re.search(r"quote\s+([A-Za-z\.\-]{1,10})", t)
    if m:
        return "GLOBAL_QUOTE", {"symbol": m.group(1).upper()}

    # News for ticker
    m = re.search(r"news\s+([A-Za-z\.\-]{1,10})", t)
    if m:
        return "NEWS_SENTIMENT", {"tickers": m.group(1).upper(), "limit": 20}

    # Search symbol
    m = re.search(r"search\s+(.+)$", t)
    if m:
        return "SYMBOL_SEARCH", {"keywords": m.group(1)}

    # Technicals: rsi AAPL daily 14
    m = re.search(r"rsi\s+([A-Za-z\.\-]{1,10})(?:\s+(1min|5min|15min|30min|60min|daily|weekly|monthly))?(?:\s+(\d{1,3}))?", t)
    if m:
        symbol = m.group(1).upper()
        interval = m.group(2) or "daily"
        period = int(m.group(3) or 14)
        return "RSI", {"symbol": symbol, "interval": interval, "time_period": period, "series_type": "close"}

    # Fallback: try market status
    if "market status" in t or "status" in t:
        return "MARKET_STATUS", {}

    # General news questions (use Gemini AI)
    news_patterns = ["latest news", "market news", "financial news", "what's happening", "market update", "economic news"]
    if any(pattern in t for pattern in news_patterns):
        return "GEMINI_NEWS", {"query": text}
    
    # Financial education questions (use Gemini AI)
    education_patterns = [
        "what is", "how does", "explain", "define", "compound interest", "diversification", 
        "portfolio", "investment strategy", "risk management", "financial planning",
        "budgeting", "saving", "retirement", "401k", "ira", "etf", "mutual fund",
        "dividend", "bond", "cryptocurrency", "inflation", "recession"
    ]
    if any(pattern in t for pattern in education_patterns):
        return "GEMINI_EDUCATION", {"query": text}
    
    # General financial advice questions (use Gemini AI)
    advice_patterns = [
        "should i invest", "how to invest", "investment advice", "financial advice",
        "best stocks", "good investment", "how much to invest", "when to buy",
        "market strategy", "investment tips"
    ]
    if any(pattern in t for pattern in advice_patterns):
        return "GEMINI_EDUCATION", {"query": text}

    # Handle greetings and casual conversation
    greeting_patterns = ["hi", "hello", "hey", "hiii", "howdy", "good morning", "good afternoon", "good evening"]
    if any(pattern in t for pattern in greeting_patterns):
        return "GREETING", {"type": "greeting"}
    
    # Handle help requests
    help_patterns = ["help", "what can you do", "how do you work", "commands"]
    if any(pattern in t for pattern in help_patterns):
        return "GREETING", {"type": "help"}

    # Final fallback: Use Gemini AI for ANY query that doesn't match specific patterns
    # This ensures all questions get intelligent responses
    return "GEMINI_EDUCATION", {"query": text}


def _maybe_parse_text_payload(result: Any) -> Any:
    # Alpha MCP often returns { content: [ { type: 'text', text: '...json...' } ] }
    try:
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            for item in result["content"]:
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    return json.loads(item["text"])  # may raise
    except Exception:
        return result
    return result


def _format_table(rows: List[Dict[str, Any]], title: str, cols: List[str]) -> str:
    if not rows:
        return f"### {title}\nNo data."
    header = " | ".join(cols)
    sep = " | ".join(["---"] * len(cols))
    lines = [f"### {title}", f"{header}", f"{sep}"]
    for r in rows:
        values = []
        for c in cols:
            v = r.get(c)
            values.append(str(v) if v is not None else "")
        lines.append(" | ".join(values))
    return "\n".join(lines)


def summarize_result(tool: str, result: Any, params: Dict[str, Any] = None) -> Any:
    # Handle greeting and help responses (no MCP call needed)
    if tool == "GREETING":
        greeting_type = params.get("type", "greeting") if params else "greeting"
        
        if greeting_type == "greeting":
            return {
                "markdown": "👋 **Hello!** I'm your AI Financial Assistant. I can help you with:\n\n" +
                           "• **Stock data**: Ask for quotes, price history, or charts\n" +
                           "• **Market insights**: Top gainers, losers, most active stocks\n" +
                           "• **Company news**: Get latest news for any stock\n" +
                           "• **Technical analysis**: RSI, moving averages, and more\n\n" +
                           "Try asking: *'What are the top gainers today?'* or *'Give me Apple stock quote'*"
            }
        elif greeting_type == "help":
            return {
                "markdown": "🤖 **AI Financial Assistant Help**\n\n" +
                           "**Stock Queries:**\n" +
                           "• `quote AAPL` - Get current stock price\n" +
                           "• `AAPL price history` - View price charts\n" +
                           "• `daily AAPL` - Daily stock data\n\n" +
                           "**Market Data:**\n" +
                           "• `top gainers` - Best performing stocks\n" +
                           "• `top losers` - Worst performing stocks\n" +
                           "• `most active stocks` - High volume stocks\n\n" +
                           "**News & Analysis:**\n" +
                           "• `news AAPL` - Company news\n" +
                           "• `search Tesla` - Find stock symbols\n" +
                           "• `RSI AAPL daily` - Technical indicators"
            }
        else:  # unknown
            return {
                "markdown": "🤔 I'm not sure how to help with that. I specialize in financial data and market insights.\n\n" +
                           "Try asking about:\n" +
                           "• Stock prices and quotes\n" +
                           "• Market trends and top performers\n" +
                           "• Company news and analysis\n\n" +
                           "For example: *'Show me Apple stock price'* or *'What are today's top gainers?'*"
            }
    
    # Handle Gemini AI responses (no MCP call needed)
    if tool == "GEMINI_NEWS":
        # This will be handled in the query endpoint with actual Gemini call
        return {"markdown": result}
    
    if tool == "GEMINI_EDUCATION":
        # This will be handled in the query endpoint with actual Gemini call
        return {"markdown": result}
    
    parsed = _maybe_parse_text_payload(result)

    if tool == "TOP_GAINERS_LOSERS" and isinstance(parsed, dict):
        gainers = parsed.get("top_gainers", [])[:10]
        losers = parsed.get("top_losers", [])[:10]
        active = parsed.get("most_actively_traded", [])[:10]

        # Normalize keys to consistent set
        def normalize(rs: List[Dict[str, Any]]):
            out = []
            for r in rs:
                out.append(
                    {
                        "ticker": r.get("ticker"),
                        "price": r.get("price"),
                        "change_%": r.get("change_percentage"),
                        "change": r.get("change_amount"),
                        "volume": r.get("volume"),
                    }
                )
            return out

        # Check if there's a filter parameter to show only specific data
        filter_type = params.get("_filter", "all") if params else "all"
        
        tables = []
        if filter_type == "gainers":
            tables.append(_format_table(normalize(gainers), "Top Gainers", ["ticker", "price", "change_%", "change", "volume"]))
        elif filter_type == "losers":
            tables.append(_format_table(normalize(losers), "Top Losers", ["ticker", "price", "change_%", "change", "volume"]))
        elif filter_type == "active":
            tables.append(_format_table(normalize(active), "Most Active", ["ticker", "price", "change_%", "change", "volume"]))
        else:  # filter_type == "all" or default
            tables.append(_format_table(normalize(gainers), "Top Gainers", ["ticker", "price", "change_%", "change", "volume"]))
            tables.append(_format_table(normalize(losers), "Top Losers", ["ticker", "price", "change_%", "change", "volume"]))
            tables.append(_format_table(normalize(active), "Most Active", ["ticker", "price", "change_%", "change", "volume"]))

        md = "\n\n".join(tables)
        return {"raw": parsed, "markdown": md, "last_updated": parsed.get("last_updated")}

    if tool == "NEWS_SENTIMENT" and isinstance(parsed, dict):
        # Normalize Alpha's news payload
        articles = []
        for a in parsed.get("feed", []) or parsed.get("data", []):
            articles.append(
                {
                    "title": a.get("title"),
                    "url": a.get("url"),
                    "time_published": a.get("time_published") or a.get("published_at"),
                    "source": a.get("source") or a.get("source_domain"),
                    "summary": a.get("summary") or a.get("snippet"),
                    "ticker_sentiment": a.get("ticker_sentiment"),
                }
            )
        return {"articles": articles, "raw": parsed}

    # Handle time series data
    if tool.startswith("TIME_SERIES_") and isinstance(parsed, dict):
        symbol = parsed.get("Meta Data", {}).get("2. Symbol", "Unknown")
        last_refreshed = parsed.get("Meta Data", {}).get("3. Last Refreshed", "Unknown")
        
        # Get the time series data key (varies by API)
        time_series_key = None
        for key in parsed.keys():
            if "Time Series" in key or "Weekly" in key or "Monthly" in key:
                time_series_key = key
                break
        
        if time_series_key and time_series_key in parsed:
            time_series = parsed[time_series_key]
            
            # Convert to list and sort by date (most recent first)
            data_points = []
            for date, values in time_series.items():
                data_points.append({
                    "date": date,
                    "open": values.get("1. open", values.get("open")),
                    "high": values.get("2. high", values.get("high")),
                    "low": values.get("3. low", values.get("low")),
                    "close": values.get("4. close", values.get("close")),
                    "volume": values.get("5. volume", values.get("volume", "N/A"))
                })
            
            # Sort by date descending and take first 10 entries
            data_points.sort(key=lambda x: x["date"], reverse=True)
            recent_data = data_points[:10]
            
            # Calculate performance metrics if we have enough data
            performance_summary = ""
            if len(data_points) >= 2:
                latest_close = float(data_points[0]["close"])
                oldest_close = float(data_points[-1]["close"])
                change = latest_close - oldest_close
                change_pct = (change / oldest_close) * 100
                performance_summary = f"\n**Performance Summary:**\n- Latest Price: ${latest_close:.2f}\n- Period Change: ${change:.2f} ({change_pct:+.2f}%)\n"
            
            # Format as markdown table
            table = _format_table(recent_data, f"{symbol} - Recent {time_series_key}", 
                                ["date", "open", "high", "low", "close", "volume"])
            
            md = f"### {symbol} Time Series Data\n**Last Refreshed:** {last_refreshed}{performance_summary}\n\n{table}"
            
            return {"raw": parsed, "markdown": md, "symbol": symbol, "last_refreshed": last_refreshed, "data_points": recent_data}

    # Handle bulk quotes
    if tool == "REALTIME_BULK_QUOTES" and isinstance(parsed, dict):
        quotes = []
        # Handle different possible response formats
        quote_data = parsed.get("Global Quote", parsed.get("quotes", parsed))
        
        if isinstance(quote_data, list):
            for quote in quote_data:
                quotes.append({
                    "symbol": quote.get("01. symbol", quote.get("symbol")),
                    "price": quote.get("05. price", quote.get("price")),
                    "change": quote.get("09. change", quote.get("change")),
                    "change_%": quote.get("10. change percent", quote.get("change_percent")),
                    "volume": quote.get("06. volume", quote.get("volume"))
                })
        elif isinstance(quote_data, dict):
            # Single quote response
            quotes.append({
                "symbol": quote_data.get("01. symbol", quote_data.get("symbol")),
                "price": quote_data.get("05. price", quote_data.get("price")),
                "change": quote_data.get("09. change", quote_data.get("change")),
                "change_%": quote_data.get("10. change percent", quote_data.get("change_percent")),
                "volume": quote_data.get("06. volume", quote_data.get("volume"))
            })
        
        if quotes:
            table = _format_table(quotes, "Real-time Quotes", ["symbol", "price", "change", "change_%", "volume"])
            return {"raw": parsed, "markdown": table, "quotes": quotes}

    # Handle single stock quote (GLOBAL_QUOTE)
    if tool == "GLOBAL_QUOTE" and isinstance(parsed, dict):
        # Check if we have content with CSV-like text data
        if "content" in parsed and isinstance(parsed["content"], list):
            for item in parsed["content"]:
                if item.get("type") == "text":
                    csv_text = item.get("text", "")
                    lines = csv_text.strip().split('\n')
                    if len(lines) >= 2:  # Header + data
                        headers = lines[0].split(',')
                        data = lines[1].split(',')
                        
                        if len(headers) == len(data):
                            # Create a dictionary from CSV data
                            stock_data = dict(zip(headers, data))
                            
                            # Extract and format the data
                            symbol = stock_data.get('symbol', 'N/A')
                            price = float(stock_data.get('price', 0))
                            open_price = float(stock_data.get('open', 0))
                            high = float(stock_data.get('high', 0))
                            low = float(stock_data.get('low', 0))
                            volume = int(stock_data.get('volume', 0))
                            change = float(stock_data.get('change', 0))
                            change_percent = stock_data.get('changePercent', '0%').replace('%', '')
                            previous_close = float(stock_data.get('previousClose', 0))
                            latest_day = stock_data.get('latestDay', 'N/A')
                            
                            # Determine if stock is up or down
                            trend_emoji = "📈" if change >= 0 else "📉"
                            change_color = "🟢" if change >= 0 else "🔴"
                            
                            # Format the markdown response
                            markdown = f"""## {trend_emoji} {symbol} Stock Quote
                            
**Current Price:** ${price:.2f} {change_color}

**Daily Performance:**
- **Change:** ${change:+.2f} ({change_percent:+}%)
- **Previous Close:** ${previous_close:.2f}

**Trading Range:**
- **Open:** ${open_price:.2f}
- **High:** ${high:.2f}
- **Low:** ${low:.2f}

**Volume:** {volume:,} shares

**Last Updated:** {latest_day}

---
*Data provided by Alpha Vantage*"""
                            
                            return {
                                "raw": parsed,
                                "markdown": markdown,
                                "symbol": symbol,
                                "price": price,
                                "change": change,
                                "change_percent": float(change_percent),
                                "volume": volume,
                                "last_updated": latest_day
                            }
        
        # Fallback if parsing fails
        return {
            "raw": parsed,
            "markdown": f"📊 **Stock Quote**\n\nReceived data for stock quote, but formatting is not available.",
            "error": "Could not parse stock quote data"
        }

    return parsed


def create_app() -> FastAPI:
    app = FastAPI(title="Alpha Vantage MCP Proxy", version="0.1.0")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Include Marketaux news router
    try:
        from .marketaux import router as marketaux_router, cleanup_marketaux  # type: ignore
        app.include_router(marketaux_router, prefix="/api")
    except Exception as e:
        print(f"Warning: Could not load Marketaux router: {e}")

    @app.on_event("startup")
    async def _startup() -> None:
        # Load environment variables from a local .env file if present
        load_dotenv()
        url = os.getenv(ALPHA_VANTAGE_MCP_URL_ENV)
        if not url:
            raise RuntimeError(f"Set {ALPHA_VANTAGE_MCP_URL_ENV} to the MCP connection URL")
        app.state.http = httpx.AsyncClient(timeout=30)
        app.state.mcp = MCPClient(url, client=app.state.http)
        
        # Initialize Gemini AI client
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if gemini_api_key:
            app.state.gemini = GeminiClient(gemini_api_key)
        else:
            print("Warning: GEMINI_API_KEY not found. AI-powered features will be disabled.")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        client: httpx.AsyncClient = app.state.http
        await client.aclose()
        # Cleanup Marketaux client
        try:
            from .marketaux import cleanup_marketaux  # type: ignore
            await cleanup_marketaux()
        except Exception:
            pass

    @app.options("/query")
    async def query_options():
        return {"message": "OK"}
    
    @app.post("/query")
    async def query(req: QueryRequest) -> Dict[str, Any]:
        tool, params = pick_tool_for_message(req.message)
        
        # Handle greeting/help responses without MCP call
        if tool == "GREETING":
            return {
                "tool": tool,
                "params": params,
                "result": summarize_result(tool, None, params),
            }
        
        # Handle Gemini AI responses
        if tool in ["GEMINI_NEWS", "GEMINI_EDUCATION"]:
            if not hasattr(app.state, 'gemini'):
                return {
                    "tool": tool,
                    "params": params,
                    "result": {"markdown": "❌ AI features are currently unavailable. Please check the server configuration."},
                }
            
            query = params.get("query", req.message)
            if tool == "GEMINI_NEWS":
                ai_response = await app.state.gemini.get_financial_news_summary(query)
            else:  # GEMINI_EDUCATION
                ai_response = await app.state.gemini.get_financial_education(query)
            
            return {
                "tool": tool,
                "params": params,
                "result": summarize_result(tool, ai_response, params),
            }
        
        # Separate internal parameters (prefixed with _) from MCP parameters
        mcp_params = {k: v for k, v in params.items() if not k.startswith('_')}
        
        result = await app.state.mcp.call_tool(tool, mcp_params)
        return {
            "tool": tool,
            "params": params,
            "result": summarize_result(tool, result, params),
        }

    @app.get("/stream")
    async def stream(message: str, interval: float = 15.0) -> StreamingResponse:
        tool, params = pick_tool_for_message(message)

        async def event_gen():
            while True:
                try:
                    # Separate internal parameters from MCP parameters
                    mcp_params = {k: v for k, v in params.items() if not k.startswith('_')}
                    res = await app.state.mcp.call_tool(tool, mcp_params)
                    summarized = summarize_result(tool, res, params)
                    data = json.dumps({"tool": tool, "params": params, "result": summarized})
                    yield f"data: {data}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(max(3.0, interval))

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    @app.get("/ui")
    async def ui() -> HTMLResponse:
        html = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Alpha Vantage MCP Tester</title>
    <script src=\"https://cdn.jsdelivr.net/npm/marked/marked.min.js\"></script>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; }
      .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
      input[type=text] { width: 360px; padding: 8px; }
      button { padding: 8px 12px; cursor: pointer; }
      pre { background: #111; color: #f5f5f5; padding: 12px; overflow: auto; }
      #md { border: 1px solid #ddd; padding: 12px; border-radius: 6px; }
      .muted { color: #666; font-size: 12px; }
    </style>
  </head>
  <body>
    <h2>Alpha Vantage MCP Tester</h2>
    <div class=\"row\">
      <input id=\"msg\" type=\"text\" value=\"top performers\" />
      <button id=\"send\">Query once</button>
      <button id=\"start\">Start stream</button>
      <button id=\"stop\">Stop stream</button>
      <span class=\"muted\">Tip: try \'past 1 week performance of AAPL\', \'daily MSFT\', \'intraday TSLA 5min\', \'quotes AAPL,MSFT,GOOGL\'</span>
    </div>
    <h3>Markdown</h3>
    <div id=\"md\"></div>
    <h3>JSON</h3>
    <pre id=\"json\"></pre>

    <script>
      const md = document.getElementById('md');
      const jsonEl = document.getElementById('json');
      let es = null;

      function render(data){
        jsonEl.textContent = JSON.stringify(data, null, 2);
        const markdown = data?.result?.markdown || 'No markdown for this tool.';
        md.innerHTML = window.marked.parse(markdown);
      }

      document.getElementById('send').onclick = async () => {
        const message = document.getElementById('msg').value;
        const res = await fetch('/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message }) });
        const data = await res.json();
        render(data);
      };

      document.getElementById('start').onclick = () => {
        const message = encodeURIComponent(document.getElementById('msg').value);
        if (es) es.close();
        es = new EventSource(`/stream?message=${message}&interval=15`);
        es.onmessage = (ev) => {
          try { render(JSON.parse(ev.data)); } catch {}
        };
      };

      document.getElementById('stop').onclick = () => { if (es) { es.close(); es = null; } };
    </script>
  </body>
</html>
"""
        return HTMLResponse(content=html)

    # News endpoints moved to app/news.py

    @app.get("/realtime/{symbol}")
    async def get_realtime_stock(symbol: str):
        """Get real-time stock data for a symbol"""
        try:
            result = await app.state.mcp.call_tool("GLOBAL_QUOTE", {"symbol": symbol.upper()})
            
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
                if isinstance(content, list) and len(content) > 0:
                    # Parse the text content to extract stock data
                    lines = content[0].get("text", "").split('\n')
                    data = {}
                    for line in lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            data[key.strip()] = value.strip()
                    
                    # Extract numeric values
                    try:
                        current_price = float(data.get("05. price", "0").replace("$", ""))
                        change = float(data.get("09. change", "0"))
                        percent_change = float(data.get("10. change percent", "0").replace("%", ""))
                        high = float(data.get("03. high", "0"))
                        low = float(data.get("04. low", "0"))
                        open_price = float(data.get("02. open", "0"))
                        previous_close = float(data.get("08. previous close", "0"))
                        
                        return {
                            "error": None,
                            "data": {
                                "current_price": current_price,
                                "change": change,
                                "percent_change": percent_change,
                                "high": high,
                                "low": low,
                                "open": open_price,
                                "previous_close": previous_close,
                                "symbol": symbol.upper()
                            }
                        }
                    except (ValueError, KeyError) as e:
                        return {"error": f"Failed to parse stock data: {str(e)}"}
                        
            return {"error": "No data available for this symbol"}
            
        except Exception as e:
            return {"error": f"Failed to fetch stock data: {str(e)}"}

    @app.get("/")
    async def root() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


