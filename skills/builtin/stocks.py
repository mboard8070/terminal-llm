"""Stock price skill using free Yahoo Finance API."""

import requests
from skills import skill


@skill(
    name="stock",
    description="Get stock prices, quotes, and basic financial data",
    version="1.0.0",
    author="MAUDE",
    triggers=["stock", "stocks", "price", "ticker", "market", "share"],
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., 'AAPL', 'NVDA', 'GOOGL')"
            },
            "action": {
                "type": "string",
                "enum": ["quote", "info", "compare"],
                "description": "Action to perform (default: quote)",
                "default": "quote"
            },
            "symbols": {
                "type": "string",
                "description": "Comma-separated symbols for compare action (e.g., 'AAPL,NVDA,MSFT')"
            }
        },
        "required": ["symbol"]
    }
)
def stock(symbol: str, action: str = "quote", symbols: str = None) -> str:
    """Get stock information."""

    if action == "compare" and symbols:
        return _compare_stocks(symbols)
    elif action == "info":
        return _get_stock_info(symbol)
    else:
        return _get_stock_quote(symbol)


def _get_stock_quote(symbol: str) -> str:
    """Get current stock quote."""
    try:
        symbol = symbol.upper().strip()

        # Use Yahoo Finance API (no key required)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        if "chart" not in data or not data["chart"]["result"]:
            return f"Symbol '{symbol}' not found. Check the ticker symbol."

        result = data["chart"]["result"][0]
        meta = result["meta"]

        price = meta.get("regularMarketPrice", 0)
        prev_close = meta.get("previousClose", 0)
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        # Determine direction
        direction = "🟢 +" if change >= 0 else "🔴 "

        # Get additional data from indicators
        quote = result.get("indicators", {}).get("quote", [{}])[0]

        output = [
            f"{meta.get('shortName', symbol)} ({symbol})",
            f"─" * 30,
            f"Price: ${price:.2f}",
            f"Change: {direction}{change:.2f} ({change_pct:+.2f}%)",
            f"Previous Close: ${prev_close:.2f}",
        ]

        # Add market status
        market_state = meta.get("marketState", "UNKNOWN")
        if market_state == "REGULAR":
            output.append("Market: Open 🟢")
        elif market_state == "CLOSED":
            output.append("Market: Closed 🔴")
        elif market_state in ("PRE", "PREPRE"):
            output.append("Market: Pre-market 🟡")
        elif market_state in ("POST", "POSTPOST"):
            output.append("Market: After-hours 🟡")

        # Add day range if available
        high = meta.get("regularMarketDayHigh")
        low = meta.get("regularMarketDayLow")
        if high and low:
            output.append(f"Day Range: ${low:.2f} - ${high:.2f}")

        # Add 52-week range
        high52 = meta.get("fiftyTwoWeekHigh")
        low52 = meta.get("fiftyTwoWeekLow")
        if high52 and low52:
            output.append(f"52-Week Range: ${low52:.2f} - ${high52:.2f}")

        return "\n".join(output)

    except requests.exceptions.Timeout:
        return "Error: Request timed out. Try again."
    except requests.exceptions.RequestException as e:
        return f"Error fetching stock data: {e}"
    except Exception as e:
        return f"Error: {e}"


def _get_stock_info(symbol: str) -> str:
    """Get detailed stock information."""
    try:
        symbol = symbol.upper().strip()

        # Use Yahoo Finance quoteSummary API
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
        params = {"modules": "summaryProfile,summaryDetail,price"}
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()

        if "quoteSummary" not in data or not data["quoteSummary"]["result"]:
            return f"Symbol '{symbol}' not found."

        result = data["quoteSummary"]["result"][0]
        price_data = result.get("price", {})
        summary = result.get("summaryDetail", {})
        profile = result.get("summaryProfile", {})

        output = [
            f"{price_data.get('shortName', symbol)} ({symbol})",
            f"─" * 40,
        ]

        # Price info
        reg_price = price_data.get("regularMarketPrice", {}).get("raw", 0)
        if reg_price:
            output.append(f"Price: ${reg_price:.2f}")

        # Company info
        if profile.get("sector"):
            output.append(f"Sector: {profile['sector']}")
        if profile.get("industry"):
            output.append(f"Industry: {profile['industry']}")
        if profile.get("country"):
            output.append(f"Country: {profile['country']}")
        if profile.get("website"):
            output.append(f"Website: {profile['website']}")

        output.append("")
        output.append("Key Metrics:")

        # Market cap
        market_cap = price_data.get("marketCap", {}).get("raw", 0)
        if market_cap:
            if market_cap >= 1e12:
                output.append(f"  Market Cap: ${market_cap/1e12:.2f}T")
            elif market_cap >= 1e9:
                output.append(f"  Market Cap: ${market_cap/1e9:.2f}B")
            else:
                output.append(f"  Market Cap: ${market_cap/1e6:.2f}M")

        # P/E ratio
        pe = summary.get("trailingPE", {}).get("raw")
        if pe:
            output.append(f"  P/E Ratio: {pe:.2f}")

        # Dividend
        div_yield = summary.get("dividendYield", {}).get("raw")
        if div_yield:
            output.append(f"  Dividend Yield: {div_yield*100:.2f}%")

        # Volume
        volume = summary.get("volume", {}).get("raw", 0)
        avg_volume = summary.get("averageVolume", {}).get("raw", 0)
        if volume:
            output.append(f"  Volume: {volume:,}")
        if avg_volume:
            output.append(f"  Avg Volume: {avg_volume:,}")

        # Beta
        beta = summary.get("beta", {}).get("raw")
        if beta:
            output.append(f"  Beta: {beta:.2f}")

        return "\n".join(output)

    except Exception as e:
        return f"Error fetching stock info: {e}"


def _compare_stocks(symbols_str: str) -> str:
    """Compare multiple stocks."""
    try:
        symbols = [s.strip().upper() for s in symbols_str.split(",")]
        if len(symbols) < 2:
            return "Please provide at least 2 symbols separated by commas."
        if len(symbols) > 5:
            symbols = symbols[:5]  # Limit to 5

        results = []
        for symbol in symbols:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {"User-Agent": "Mozilla/5.0"}

            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()

            if "chart" in data and data["chart"]["result"]:
                meta = data["chart"]["result"][0]["meta"]
                price = meta.get("regularMarketPrice", 0)
                prev_close = meta.get("previousClose", 0)
                change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0

                results.append({
                    "symbol": symbol,
                    "name": meta.get("shortName", symbol),
                    "price": price,
                    "change_pct": change_pct
                })

        if not results:
            return "Could not fetch data for any of the symbols."

        # Sort by change percentage
        results.sort(key=lambda x: x["change_pct"], reverse=True)

        output = ["Stock Comparison:", "─" * 50]
        for r in results:
            direction = "🟢" if r["change_pct"] >= 0 else "🔴"
            output.append(
                f"  {direction} {r['symbol']:6} ${r['price']:>10.2f}  {r['change_pct']:>+7.2f}%"
            )

        return "\n".join(output)

    except Exception as e:
        return f"Error comparing stocks: {e}"


@skill(
    name="crypto",
    description="Get cryptocurrency prices",
    version="1.0.0",
    author="MAUDE",
    triggers=["crypto", "bitcoin", "btc", "ethereum", "eth", "coin"],
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Crypto symbol (e.g., 'BTC', 'ETH', 'SOL')",
                "default": "BTC"
            }
        }
    }
)
def crypto(symbol: str = "BTC") -> str:
    """Get cryptocurrency price."""
    try:
        symbol = symbol.upper().strip()

        # Map common names to Yahoo Finance format
        crypto_map = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
            "ADA": "ADA-USD",
            "DOGE": "DOGE-USD",
            "XRP": "XRP-USD",
            "DOT": "DOT-USD",
            "AVAX": "AVAX-USD",
            "MATIC": "MATIC-USD",
            "LINK": "LINK-USD",
        }

        yahoo_symbol = crypto_map.get(symbol, f"{symbol}-USD")

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()

        if "chart" not in data or not data["chart"]["result"]:
            return f"Crypto '{symbol}' not found. Try: BTC, ETH, SOL, ADA, DOGE, XRP"

        meta = data["chart"]["result"][0]["meta"]

        price = meta.get("regularMarketPrice", 0)
        prev_close = meta.get("previousClose", 0)
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        direction = "🟢 +" if change >= 0 else "🔴 "

        output = [
            f"{meta.get('shortName', symbol)} ({symbol})",
            f"─" * 30,
            f"Price: ${price:,.2f}",
            f"24h Change: {direction}{change:,.2f} ({change_pct:+.2f}%)",
        ]

        high = meta.get("regularMarketDayHigh")
        low = meta.get("regularMarketDayLow")
        if high and low:
            output.append(f"24h Range: ${low:,.2f} - ${high:,.2f}")

        return "\n".join(output)

    except Exception as e:
        return f"Error fetching crypto data: {e}"
