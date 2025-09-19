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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

ALPHA_VANTAGE_MCP_URL_ENV = "ALPHA_VANTAGE_MCP_URL"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"


class QueryRequest(BaseModel):
    message: str


class AIQueryParser:
    """AI-powered query parser using Gemini to understand user intent and generate MCP tool calls"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Define available MCP tools and their parameters
        self.tool_definitions = {
            "TIME_SERIES_INTRADAY": {
                "description": "Get intraday OHLCV data with intervals (1min, 5min, 15min, 30min, 60min)",
                "required_params": ["symbol", "interval"],
                "optional_params": ["outputsize"],
                "example": {"symbol": "AAPL", "interval": "5min", "outputsize": "compact"}
            },
            "TIME_SERIES_DAILY": {
                "description": "Get daily OHLCV data covering 20+ years",
                "required_params": ["symbol"],
                "optional_params": ["outputsize"],
                "example": {"symbol": "AAPL", "outputsize": "compact"}
            },
            "TIME_SERIES_DAILY_ADJUSTED": {
                "description": "Get daily adjusted OHLCV with split/dividend events",
                "required_params": ["symbol"],
                "optional_params": ["outputsize"],
                "example": {"symbol": "AAPL", "outputsize": "compact"}
            },
            "TIME_SERIES_WEEKLY": {
                "description": "Get weekly time series (last trading day of week)",
                "required_params": ["symbol"],
                "optional_params": [],
                "example": {"symbol": "AAPL"}
            },
            "TIME_SERIES_WEEKLY_ADJUSTED": {
                "description": "Get weekly adjusted time series with dividends",
                "required_params": ["symbol"],
                "optional_params": [],
                "example": {"symbol": "AAPL"}
            },
            "TIME_SERIES_MONTHLY": {
                "description": "Get monthly time series (last trading day of month)",
                "required_params": ["symbol"],
                "optional_params": [],
                "example": {"symbol": "AAPL"}
            },
            "TIME_SERIES_MONTHLY_ADJUSTED": {
                "description": "Get monthly adjusted time series with dividends",
                "required_params": ["symbol"],
                "optional_params": [],
                "example": {"symbol": "AAPL"}
            },
            "GLOBAL_QUOTE": {
                "description": "Get latest price and volume for a ticker",
                "required_params": ["symbol"],
                "optional_params": [],
                "example": {"symbol": "AAPL"}
            },
            "REALTIME_BULK_QUOTES": {
                "description": "Get realtime quotes for up to 100 symbols",
                "required_params": ["symbols"],
                "optional_params": [],
                "example": {"symbols": "AAPL,MSFT,GOOGL"}
            },
            "SYMBOL_SEARCH": {
                "description": "Search for symbols by keywords",
                "required_params": ["keywords"],
                "optional_params": [],
                "example": {"keywords": "Apple"}
            },
            "MARKET_STATUS": {
                "description": "Get current market status worldwide",
                "required_params": [],
                "optional_params": [],
                "example": {}
            },
            "TOP_GAINERS_LOSERS": {
                "description": "Get top gainers, losers, and most active stocks",
                "required_params": [],
                "optional_params": [],
                "example": {}
            },
            "NEWS_SENTIMENT": {
                "description": "Get news and sentiment analysis for stocks",
                "required_params": ["tickers"],
                "optional_params": ["limit"],
                "example": {"tickers": "AAPL", "limit": 20}
            },
            "RSI": {
                "description": "Get Relative Strength Index technical indicator",
                "required_params": ["symbol", "interval", "time_period", "series_type"],
                "optional_params": [],
                "example": {"symbol": "AAPL", "interval": "daily", "time_period": 14, "series_type": "close"}
            }
        }
    
    async def parse_query(self, user_query: str) -> tuple[str, Dict[str, Any]]:
        """Use AI to parse user query and return appropriate tool and parameters"""
        
        prompt = f"""
You are an expert financial data API assistant. Analyze the user query and determine the best MCP tool to use and its parameters.

Available MCP Tools:
{json.dumps(self.tool_definitions, indent=2)}

User Query: "{user_query}"

Rules:
1. For performance queries like "past 1 week performance of AAPL":
   - Use TIME_SERIES_INTRADAY for periods <= 1 week (with 60min interval)
   - Use TIME_SERIES_DAILY for periods <= 1 month
   - Use TIME_SERIES_WEEKLY for periods <= 1 year
   - Use TIME_SERIES_MONTHLY for periods > 1 year

2. For time series queries:
   - "intraday" -> TIME_SERIES_INTRADAY
   - "daily" -> TIME_SERIES_DAILY (or TIME_SERIES_DAILY_ADJUSTED if "adjusted" mentioned)
   - "weekly" -> TIME_SERIES_WEEKLY (or TIME_SERIES_WEEKLY_ADJUSTED if "adjusted" mentioned)
   - "monthly" -> TIME_SERIES_MONTHLY (or TIME_SERIES_MONTHLY_ADJUSTED if "adjusted" mentioned)

3. For quotes:
   - Single symbol -> GLOBAL_QUOTE
   - Multiple symbols (comma-separated) -> REALTIME_BULK_QUOTES

4. Extract stock symbols and convert to uppercase
5. Set appropriate intervals for intraday data (1min, 5min, 15min, 30min, 60min)
6. Use "compact" for outputsize when available

Respond with ONLY a JSON object in this format:
{{
    "tool": "TOOL_NAME",
    "params": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "reasoning": "Brief explanation of why this tool and these parameters were chosen"
}}
"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()
            
            parsed_response = json.loads(response_text)
            
            tool = parsed_response.get("tool")
            params = parsed_response.get("params", {})
            
            # Validate tool exists
            if tool not in self.tool_definitions:
                raise ValueError(f"Unknown tool: {tool}")
            
            return tool, params
            
        except Exception as e:
            print(f"AI parsing failed: {e}")
            # Fallback to regex-based parsing
            return self._fallback_parse(user_query)
    
    def _fallback_parse(self, text: str) -> tuple[str, Dict[str, Any]]:
        """Fallback to original regex-based parsing if AI fails"""
        return pick_tool_for_message(text)


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

    # Top gainers/losers/most active
    if (
        "top" in t
        or "top performers" in t
        or "top performance" in t
        or "gainers" in t
        or "most active" in t
        or "growing" in t
    ):
        return "TOP_GAINERS_LOSERS", {}

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

    # Final fallback: treat as top gainers to emulate example behavior
    return "TOP_GAINERS_LOSERS", {}


def _parse_csv_data(csv_text: str) -> List[Dict[str, Any]]:
    """Parse CSV data into list of dictionaries"""
    lines = [line.strip() for line in csv_text.strip().split('\n') if line.strip()]
    if not lines:
        return []
    
    # Get headers - handle carriage returns
    headers = [h.strip().replace('\r', '') for h in lines[0].split(',')]
    
    # Parse data rows
    data = []
    for line in lines[1:]:
        if line.strip():
            # Handle carriage returns and split by comma
            values = [v.strip().replace('\r', '') for v in line.split(',')]
            if len(values) == len(headers):
                row = dict(zip(headers, values))
                data.append(row)
    
    return data


def _maybe_parse_text_payload(result: Any) -> Any:
    # Alpha MCP often returns { content: [ { type: 'text', text: '...json...' } ] }
    try:
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            for item in result["content"]:
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_content = item["text"]
                    
                    # Try to parse as JSON first
                    try:
                        return json.loads(text_content)
                    except json.JSONDecodeError:
                        # If JSON parsing fails, check if it's CSV data
                        if ("," in text_content and "\n" in text_content and 
                            any(header in text_content.lower() for header in ["timestamp", "date", "symbol", "open", "high", "low", "close", "price"])):
                            csv_data = _parse_csv_data(text_content)
                            if csv_data:
                                return {"csv_data": csv_data, "raw_csv": text_content}
                        return {"raw_text": text_content}
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
    parsed = _maybe_parse_text_payload(result)

    # Handle CSV data for any tool
    if isinstance(parsed, dict) and "csv_data" in parsed:
        csv_data = parsed["csv_data"]
        
        if tool == "GLOBAL_QUOTE" and csv_data:
            # Handle single quote CSV data
            quote = csv_data[0] if csv_data else {}
            symbol = quote.get("symbol", params.get("symbol", "Unknown") if params else "Unknown")
            
            quote_info = {
                "symbol": symbol,
                "price": quote.get("price", "N/A"),
                "change": quote.get("change", "N/A"),
                "change_percent": quote.get("changePercent", "N/A"),
                "volume": quote.get("volume", "N/A"),
                "open": quote.get("open", "N/A"),
                "high": quote.get("high", "N/A"),
                "low": quote.get("low", "N/A"),
                "previous_close": quote.get("previousClose", "N/A"),
                "latest_day": quote.get("latestDay", "N/A")
            }
            
            # Format as markdown
            md = f"""### {symbol} Stock Quote
**Latest Price:** ${quote_info['price']} ({quote_info['latest_day']})
**Change:** {quote_info['change']} ({quote_info['change_percent']})
**Previous Close:** ${quote_info['previous_close']}

**Trading Data:**
- **Open:** ${quote_info['open']}
- **High:** ${quote_info['high']}
- **Low:** ${quote_info['low']}
- **Volume:** {quote_info['volume']:,} shares
"""
            
            return {"raw": parsed, "markdown": md, "quote": quote_info}
    
    # Handle raw text data
    if isinstance(parsed, dict) and "raw_text" in parsed:
        raw_text = parsed["raw_text"]
        return {"raw": parsed, "markdown": f"### Raw Data\n```\n{raw_text}\n```"}

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

        gtab = _format_table(normalize(gainers), "Top Gainers", ["ticker", "price", "change_%", "change", "volume"])
        ltab = _format_table(normalize(losers), "Top Losers", ["ticker", "price", "change_%", "change", "volume"])
        atab = _format_table(normalize(active), "Most Active", ["ticker", "price", "change_%", "change", "volume"])

        md = "\n\n".join([gtab, ltab, atab])
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
        # Handle CSV data format
        if "csv_data" in parsed:
            csv_data = parsed["csv_data"]
            
            # Convert CSV data to our standard format
            data_points = []
            for row in csv_data:
                # Handle different timestamp formats
                timestamp = row.get("timestamp", row.get("date", ""))
                if timestamp:
                    data_points.append({
                        "date": timestamp.split(" ")[0] if " " in timestamp else timestamp,
                        "time": timestamp.split(" ")[1] if " " in timestamp else "",
                        "open": row.get("open", ""),
                        "high": row.get("high", ""),
                        "low": row.get("low", ""),
                        "close": row.get("close", ""),
                        "volume": row.get("volume", "N/A")
                    })
            
            # Sort by timestamp descending and take first 20 entries for display
            data_points.sort(key=lambda x: x.get("date", "") + " " + x.get("time", ""), reverse=True)
            recent_data = data_points[:20]
            
            # Extract symbol from the query parameters or use a default
            symbol = params.get("symbol", "STOCK") if params else "STOCK"
            
            # Calculate performance metrics if we have enough data
            performance_summary = ""
            if len(data_points) >= 2:
                try:
                    latest_close = float(data_points[0]["close"])
                    oldest_close = float(data_points[-1]["close"])
                    change = latest_close - oldest_close
                    change_pct = (change / oldest_close) * 100
                    
                    # Get time range
                    latest_time = data_points[0]["date"] + (" " + data_points[0]["time"] if data_points[0]["time"] else "")
                    oldest_time = data_points[-1]["date"] + (" " + data_points[-1]["time"] if data_points[-1]["time"] else "")
                    
                    performance_summary = f"\n**Performance Summary:**\n- Latest Price: ${latest_close:.2f} ({latest_time})\n- Oldest Price: ${oldest_close:.2f} ({oldest_time})\n- Total Change: ${change:.2f} ({change_pct:+.2f}%)\n- Data Points: {len(data_points)}\n"
                except (ValueError, TypeError):
                    performance_summary = f"\n**Data Summary:**\n- Total Data Points: {len(data_points)}\n"
            
            # Format as markdown table (show recent data with time if available)
            display_cols = ["date", "time", "open", "high", "low", "close", "volume"] if recent_data and recent_data[0].get("time") else ["date", "open", "high", "low", "close", "volume"]
            table = _format_table(recent_data, f"{symbol} - Recent Time Series Data", display_cols)
            
            md = f"### {symbol} Time Series Data{performance_summary}\n\n{table}"
            
            return {"raw": parsed, "markdown": md, "symbol": symbol, "data_points": recent_data, "total_points": len(data_points)}
        
        # Handle traditional JSON format (fallback)
        else:
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

    return parsed


def create_app() -> FastAPI:
    app = FastAPI(title="Alpha Vantage MCP Proxy", version="0.1.0")
    
    # Mount static files for frontend
    try:
        app.mount("/static", StaticFiles(directory="frontend"), name="static")
    except Exception as e:
        print(f"Warning: Could not mount static files: {e}")
    
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
        gemini_key = os.getenv(GEMINI_API_KEY_ENV)
        
        if not url:
            raise RuntimeError(f"Set {ALPHA_VANTAGE_MCP_URL_ENV} to the MCP connection URL")
        if not gemini_key:
            raise RuntimeError(f"Set {GEMINI_API_KEY_ENV} to your Gemini API key")
            
        app.state.http = httpx.AsyncClient(timeout=30)
        app.state.mcp = MCPClient(url, client=app.state.http)
        app.state.ai_parser = AIQueryParser(gemini_key)

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

    @app.post("/query")
    async def query(req: QueryRequest) -> Dict[str, Any]:
        try:
            # Use AI parser for intelligent query understanding
            tool, params = await app.state.ai_parser.parse_query(req.message)
        except Exception as e:
            print(f"AI parsing failed, using fallback: {e}")
            # Fallback to regex-based parsing
            tool, params = pick_tool_for_message(req.message)
        
        try:
            result = await app.state.mcp.call_tool(tool, params)
            return {
                "tool": tool,
                "params": params,
                "result": summarize_result(tool, result, params),
                "ai_powered": True
            }
        except Exception as e:
            return {
                "tool": tool,
                "params": params,
                "error": str(e),
                "ai_powered": True
            }

    @app.get("/stream")
    async def stream(message: str, interval: float = 15.0) -> StreamingResponse:
        try:
            tool, params = await app.state.ai_parser.parse_query(message)
        except Exception as e:
            print(f"AI parsing failed in stream, using fallback: {e}")
            tool, params = pick_tool_for_message(message)

        async def event_gen():
            while True:
                try:
                    res = await app.state.mcp.call_tool(tool, params)
                    summarized = summarize_result(tool, res, params)
                    data = json.dumps({"tool": tool, "params": params, "result": summarized, "ai_powered": True})
                    yield f"data: {data}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e), 'tool': tool, 'params': params})}\n\n"
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

    @app.get("/")
    async def root() -> HTMLResponse:
        try:
            with open("simple_frontend.html", "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
            return HTMLResponse(content="<h1>Frontend not found</h1><p>Please ensure the simple_frontend.html file exists.</p>")

    @app.get("/api/status")
    async def api_status() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/debug/pattern")
    async def debug_pattern(req: QueryRequest) -> Dict[str, Any]:
        """Debug endpoint to test pattern matching without calling MCP"""
        tool, params = pick_tool_for_message(req.message)
        return {
            "message": req.message,
            "tool": tool,
            "params": params,
            "method": "regex",
            "mcp_url": os.getenv(ALPHA_VANTAGE_MCP_URL_ENV)
        }

    @app.post("/debug/ai")
    async def debug_ai(req: QueryRequest) -> Dict[str, Any]:
        """Debug endpoint to test AI parsing without calling MCP"""
        try:
            tool, params = await app.state.ai_parser.parse_query(req.message)
            return {
                "message": req.message,
                "tool": tool,
                "params": params,
                "method": "ai",
                "status": "success"
            }
        except Exception as e:
            # Try fallback
            fallback_tool, fallback_params = pick_tool_for_message(req.message)
            return {
                "message": req.message,
                "ai_error": str(e),
                "fallback_tool": fallback_tool,
                "fallback_params": fallback_params,
                "method": "ai_with_fallback",
                "status": "ai_failed"
            }

    return app


app = create_app()


