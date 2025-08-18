# tools.py
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from mcp_stock_agent import MCPServer


class StockAnalysisTools:
    """股票分析工具集"""

    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys

    async def get_stock_price(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """获取股票价格数据"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)

            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "current_price": float(hist['Close'].iloc[-1]),
                    "open": float(hist['Open'].iloc[-1]),
                    "high": float(hist['High'].iloc[-1]),
                    "low": float(hist['Low'].iloc[-1]),
                    "volume": int(hist['Volume'].iloc[-1]),
                    "price_change": float(hist['Close'].iloc[-1] - hist['Close'].iloc[-2]),
                    "price_change_pct": float(
                        (hist['Close'].iloc[-1] - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2] * 100
                        ),
                    "history": hist.to_dict('records')
                    }
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """获取财务数据"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            financials = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "roe": info.get("returnOnEquity"),
                "revenue_growth": info.get("revenueGrowth"),
                "gross_margins": info.get("grossMargins"),
                "operating_margins": info.get("operatingMargins"),
                "profit_margins": info.get("profitMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                "free_cash_flow": info.get("freeCashflow"),
                "earnings_growth": info.get("earningsGrowth"),
                "recommendation": info.get("recommendationKey", "none")
                }

            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "company_name": info.get("longName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "financials": financials
                    }
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def calculate_technical_indicators(self, symbol: str, period: str = "3mo") -> Dict[str, Any]:
        """计算技术指标"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            df = pd.DataFrame(hist)

            # 计算移动平均线
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()

            # 计算 RSI
            def calculate_rsi(data, period=14):
                delta = data.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            df['RSI'] = calculate_rsi(df['Close'])

            # 计算 MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_diff'] = df['MACD'] - df['Signal']

            # 计算布林带
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + 2 * bb_std
            df['BB_lower'] = df['BB_middle'] - 2 * bb_std

            # 支撑和阻力位
            recent_high = df['High'].tail(20).max()
            recent_low = df['Low'].tail(20).min()

            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "current_price": float(df['Close'].iloc[-1]),
                    "indicators": {
                        "rsi": float(df['RSI'].iloc[-1]),
                        "macd": float(df['MACD'].iloc[-1]),
                        "macd_signal": float(df['Signal'].iloc[-1]),
                        "ma_20": float(df['MA_20'].iloc[-1]) if not pd.isna(df['MA_20'].iloc[-1]) else None,
                        "ma_50": float(df['MA_50'].iloc[-1]) if not pd.isna(df['MA_50'].iloc[-1]) else None,
                        "ma_200": float(df['MA_200'].iloc[-1]) if not pd.isna(df['MA_200'].iloc[-1]) else None,
                        "bb_upper": float(df['BB_upper'].iloc[-1]),
                        "bb_middle": float(df['BB_middle'].iloc[-1]),
                        "bb_lower": float(df['BB_lower'].iloc[-1]),
                        "support": float(recent_low),
                        "resistance": float(recent_high)
                        },
                    "signals": self._generate_signals(df)
                    }
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_signals(self, df: pd.DataFrame) -> Dict[str, str]:
        """生成交易信号"""
        signals = {}

        # RSI 信号
        rsi = df['RSI'].iloc[-1]
        if rsi < 30:
            signals['rsi'] = "超卖 - 可能反弹"
        elif rsi > 70:
            signals['rsi'] = "超买 - 可能回调"
        else:
            signals['rsi'] = "中性"

        # MACD 信号
        if df['MACD_diff'].iloc[-1] > 0 and df['MACD_diff'].iloc[-2] <= 0:
            signals['macd'] = "金叉 - 看涨信号"
        elif df['MACD_diff'].iloc[-1] < 0 and df['MACD_diff'].iloc[-2] >= 0:
            signals['macd'] = "死叉 - 看跌信号"
        else:
            signals['macd'] = "中性"

        # 移动平均线信号
        current_price = df['Close'].iloc[-1]
        ma20 = df['MA_20'].iloc[-1]
        ma50 = df['MA_50'].iloc[-1]

        if not pd.isna(ma20) and not pd.isna(ma50):
            if current_price > ma20 > ma50:
                signals['ma'] = "多头排列 - 上升趋势"
            elif current_price < ma20 < ma50:
                signals['ma'] = "空头排列 - 下降趋势"
            else:
                signals['ma'] = "趋势不明"

        return signals

    async def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """获取新闻情绪分析"""
        # 这里使用模拟数据，实际应接入新闻 API
        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "sentiment_score": 0.65,  # -1 到 1
                "sentiment_label": "positive",
                "news_count": 15,
                "key_topics": ["earnings beat", "product launch", "market expansion"],
                "recent_news": [
                    {
                        "title": "Company beats Q3 earnings expectations",
                        "sentiment": 0.8,
                        "source": "Reuters",
                        "date": "2025-08-17"
                        }
                    ]
                }
            }

    async def screen_stocks(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """股票筛选器"""
        # 示例实现
        screener_results = {
            "success": True,
            "data": {
                "criteria": criteria,
                "matches": [
                    {"symbol": "AAPL", "score": 0.92, "metrics": {"pe": 28.5, "roe": 0.147}},
                    {"symbol": "MSFT", "score": 0.89, "metrics": {"pe": 32.1, "roe": 0.438}},
                    {"symbol": "GOOGL", "score": 0.87, "metrics": {"pe": 25.3, "roe": 0.276}}
                    ],
                "total_matches": 3
                }
            }
        return screener_results

    async def backtest_strategy(self, symbol: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """策略回测"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            df = pd.DataFrame(hist)

            # 简单的均线策略回测
            df['MA_short'] = df['Close'].rolling(window=strategy.get('short_ma', 10)).mean()
            df['MA_long'] = df['Close'].rolling(window=strategy.get('long_ma', 30)).mean()

            # 生成交易信号
            df['Signal'] = 0
            df.loc[df['MA_short'] > df['MA_long'], 'Signal'] = 1
            df.loc[df['MA_short'] < df['MA_long'], 'Signal'] = -1

            # 计算收益
            df['Returns'] = df['Close'].pct_change()
            df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']

            # 计算累计收益
            cumulative_returns = (1 + df['Strategy_Returns']).cumprod() - 1
            total_return = float(cumulative_returns.iloc[-1])

            # 计算最大回撤
            cumulative = (1 + df['Strategy_Returns']).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = float(drawdown.min())

            # 计算夏普比率
            sharpe_ratio = float(df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252))

            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "strategy": strategy,
                    "performance": {
                        "total_return": total_return,
                        "annualized_return": total_return,  # 简化计算
                        "max_drawdown": max_drawdown,
                        "sharpe_ratio": sharpe_ratio,
                        "win_rate": float((df['Strategy_Returns'] > 0).sum() / (df['Strategy_Returns'] != 0).sum()),
                        "trades_count": int((df['Signal'].diff() != 0).sum())
                        }
                    }
                }
        except Exception as e:
            return {"success": False, "error": str(e)}


# 工具注册函数
def register_tools(mcp_server: MCPServer, api_keys: Dict[str, str]):
    """注册所有工具到 MCP 服务器"""
    tools = StockAnalysisTools(api_keys)

    mcp_server.tools = {
        "get_stock_price": tools.get_stock_price,
        "get_financial_data": tools.get_financial_data,
        "calculate_technical_indicators": tools.calculate_technical_indicators,
        "get_news_sentiment": tools.get_news_sentiment,
        "screen_stocks": tools.screen_stocks,
        "backtest_strategy": tools.backtest_strategy
        }
