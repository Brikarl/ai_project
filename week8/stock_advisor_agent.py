# stock_advisor_agent.py
import json
from datetime import datetime
from typing import Dict, Any, List

from mcp_stock_agent import MCPServer


class StockAdvisorAgent:
    """股票金融建议 Agent"""

    def __init__(self, mcp_server: MCPServer):
        self.mcp_server = mcp_server
        self.conversation_history = []

    async def process_request(self, user_input: str) -> Dict[str, Any]:
        """处理用户请求"""
        # 解析用户意图
        intent = await self._parse_intent(user_input)

        # 根据意图选择合适的 prompt
        prompt_name = self._select_prompt(intent)

        # 收集所需数据
        data = await self._gather_data(intent)

        # 生成响应
        response = await self._generate_response(prompt_name, intent, data)

        # 保存对话历史
        self.conversation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "intent": intent,
                "response": response
                }
            )

        return response

    async def _parse_intent(self, user_input: str) -> Dict[str, Any]:
        """解析用户意图"""
        user_input_lower = user_input.lower()

        # 简单的意图识别（实际应使用 NLU）
        if any(word in user_input_lower for word in ["分析", "股票", "evaluate", "analysis"]):
            # 提取股票代码
            import re
            symbols = re.findall(r'\b[A-Z]{1,5}\b', user_input)
            return {
                "type": "stock_analysis",
                "symbols": symbols if symbols else [],
                "params": {}
                }

        elif any(word in user_input_lower for word in ["组合", "portfolio", "优化", "配置"]):
            return {
                "type": "portfolio_optimization",
                "params": {
                    "risk_tolerance": self._extract_risk_tolerance(user_input)
                    }
                }

        elif any(word in user_input_lower for word in ["市场", "market", "趋势", "行情"]):
            return {
                "type": "market_insight",
                "params": {}
                }

        elif any(word in user_input_lower for word in ["筛选", "screen", "找", "推荐"]):
            return {
                "type": "stock_screening",
                "params": self._extract_screening_criteria(user_input)
                }

        else:
            return {
                "type": "general_query",
                "params": {}
                }

    def _extract_risk_tolerance(self, user_input: str) -> str:
        """提取风险承受能力"""
        if any(word in user_input.lower() for word in ["保守", "conservative", "低风险"]):
            return "conservative"
        elif any(word in user_input.lower() for word in ["激进", "aggressive", "高风险"]):
            return "aggressive"
        else:
            return "moderate"

    def _extract_screening_criteria(self, user_input: str) -> Dict[str, Any]:
        """提取筛选条件"""
        criteria = {}

        # 简单的条件提取
        if "低市盈率" in user_input or "low pe" in user_input.lower():
            criteria["pe_max"] = 20

        if "高增长" in user_input or "high growth" in user_input.lower():
            criteria["revenue_growth_min"] = 0.15

        if "分红" in user_input or "dividend" in user_input.lower():
            criteria["dividend_yield_min"] = 0.02

        return criteria

    def _select_prompt(self, intent: Dict[str, Any]) -> str:
        """选择合适的 prompt"""
        intent_type = intent.get("type")

        if intent_type == "stock_analysis":
            return "stock_analysis"
        elif intent_type == "portfolio_optimization":
            return "portfolio_optimization"
        elif intent_type == "market_insight":
            return "market_insight"
        else:
            return "general_advice"

    async def _gather_data(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """收集分析所需的数据"""
        data = {}
        intent_type = intent.get("type")

        if intent_type == "stock_analysis":
            symbols = intent.get("symbols", [])
            for symbol in symbols:
                # 获取股票数据
                price_data = await self.mcp_server.tools["get_stock_price"](symbol)
                financial_data = await self.mcp_server.tools["get_financial_data"](symbol)
                technical_data = await self.mcp_server.tools["calculate_technical_indicators"](symbol)
                sentiment_data = await self.mcp_server.tools["get_news_sentiment"](symbol)

                data[symbol] = {
                    "price": price_data,
                    "financials": financial_data,
                    "technicals": technical_data,
                    "sentiment": sentiment_data
                    }

        elif intent_type == "portfolio_optimization":
            # 获取用户组合数据
            portfolio_resource = await self.mcp_server.resources["stock://portfolio/default"].read()
            data["portfolio"] = portfolio_resource["data"]

            # 获取市场数据
            market_resource = await self.mcp_server.resources["stock://market/data"].read()
            data["market"] = market_resource["data"]

        elif intent_type == "market_insight":
            # 获取市场数据
            market_resource = await self.mcp_server.resources["stock://market/data"].read()
            data["market"] = market_resource["data"]

            # 获取关键指数的技术指标
            for index in ["SPY", "QQQ"]:
                technical_data = await self.mcp_server.tools["calculate_technical_indicators"](index)
                data[f"{index}_technicals"] = technical_data

        elif intent_type == "stock_screening":
            # 执行股票筛选
            criteria = intent.get("params", {})
            screening_result = await self.mcp_server.tools["screen_stocks"](criteria)
            data["screening_result"] = screening_result

        return data

    async def _generate_response(self, prompt_name: str, intent: Dict[str, Any], data: Dict[str, Any]) -> Dict[
        str, Any]:
        """生成响应"""
        try:
            # 获取 prompt 模板
            prompt = self.mcp_server.prompts.get(prompt_name)
            if not prompt:
                return {
                    "success": False,
                    "error": "未找到合适的分析模板"
                    }

            # 准备 prompt 参数
            prompt_params = {}

            if prompt_name == "stock_analysis":
                symbols = intent.get("symbols", [])
                if symbols:
                    symbol = symbols[0]
                    prompt_params["symbol"] = symbol
                    prompt_params["analysis_type"] = intent.get("params", {}).get("analysis_type", "综合")
                    prompt_params["stock_data"] = json.dumps(data.get(symbol, {}), ensure_ascii=False, indent=2)

            elif prompt_name == "portfolio_optimization":
                prompt_params["current_portfolio"] = json.dumps(data.get("portfolio", {}), ensure_ascii=False, indent=2)
                prompt_params["risk_tolerance"] = intent.get("params", {}).get("risk_tolerance", "moderate")
                prompt_params["investment_goal"] = intent.get("params", {}).get("investment_goal", "长期增值")

            elif prompt_name == "market_insight":
                prompt_params["market_data"] = json.dumps(data, ensure_ascii=False, indent=2)

            # 格式化 prompt
            formatted_prompt = prompt["template"].format(**prompt_params)

            # 这里应该调用 LLM API，现在返回结构化的分析结果
            analysis_result = await self._perform_analysis(prompt_name, data, intent)

            return {
                "success": True,
                "intent": intent,
                "analysis": analysis_result,
                "recommendations": await self._generate_recommendations(analysis_result),
                "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
                }

    async def _perform_analysis(self, prompt_name: str, data: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
        """执行分析（模拟 LLM 分析结果）"""
        if prompt_name == "stock_analysis":
            symbols = intent.get("symbols", [])
            if symbols:
                symbol = symbols[0]
                stock_data = data.get(symbol, {})

                # 基于数据生成分析
                price_data = stock_data.get("price", {}).get("data", {})
                financial_data = stock_data.get("financials", {}).get("data", {})
                technical_data = stock_data.get("technicals", {}).get("data", {})
                sentiment_data = stock_data.get("sentiment", {}).get("data", {})

                return {
                    "symbol": symbol,
                    "rating": self._calculate_rating(financial_data, technical_data, sentiment_data),
                    "fundamental_analysis": {
                        "financial_health": self._analyze_financials(financial_data),
                        "valuation": self._analyze_valuation(financial_data),
                        "growth_potential": self._analyze_growth(financial_data)
                        },
                    "technical_analysis": {
                        "trend": self._analyze_trend(technical_data),
                        "momentum": self._analyze_momentum(technical_data),
                        "support_resistance": {
                            "support": technical_data.get("indicators", {}).get("support"),
                            "resistance": technical_data.get("indicators", {}).get("resistance")
                            }
                        },
                    "market_sentiment": {
                        "news_sentiment": sentiment_data.get("sentiment_label", "neutral"),
                        "sentiment_score": sentiment_data.get("sentiment_score", 0)
                        },
                    "risks": self._identify_risks(stock_data)
                    }

        elif prompt_name == "portfolio_optimization":
            portfolio_data = data.get("portfolio", {})
            market_data = data.get("market", {})
            risk_tolerance = intent.get("params", {}).get("risk_tolerance", "moderate")

            return {
                "current_allocation": self._analyze_current_allocation(portfolio_data),
                "optimal_allocation": self._calculate_optimal_allocation(portfolio_data, risk_tolerance, market_data),
                "rebalancing_suggestions": self._generate_rebalancing_suggestions(portfolio_data, risk_tolerance),
                "risk_analysis": self._portfolio_risk_analysis(portfolio_data)
                }

        elif prompt_name == "market_insight":
            market_data = data.get("market", {}).get("data", {})

            return {
                "market_overview": {
                    "trend": self._analyze_market_trend(market_data),
                    "breadth": self._analyze_market_breadth(market_data),
                    "volatility": market_data.get("vix", 0)
                    },
                "sector_rotation": self._analyze_sector_rotation(market_data.get("sectors", {})),
                "opportunities": self._identify_opportunities(market_data),
                "risks": self._identify_market_risks(market_data)
                }

        return {}

    def _calculate_rating(self, financial_data: Dict, technical_data: Dict, sentiment_data: Dict) -> str:
        """计算综合评级"""
        score = 0

        # 基本面评分
        financials = financial_data.get("financials", {})
        if financials.get("pe_ratio") and financials["pe_ratio"] < 25:
            score += 1
        if financials.get("roe") and financials["roe"] > 0.15:
            score += 1
        if financials.get("revenue_growth") and financials["revenue_growth"] > 0.1:
            score += 1

        # 技术面评分
        indicators = technical_data.get("indicators", {})
        if indicators.get("rsi") and 30 < indicators["rsi"] < 70:
            score += 1
        signals = technical_data.get("signals", {})
        if signals.get("macd") == "金叉 - 看涨信号":
            score += 1

        # 情绪评分
        if sentiment_data.get("sentiment_score", 0) > 0.5:
            score += 1

        # 转换为评级
        if score >= 5:
            return "强烈买入"
        elif score >= 4:
            return "买入"
        elif score >= 2:
            return "持有"
        elif score >= 1:
            return "卖出"
        else:
            return "强烈卖出"

    def _analyze_financials(self, financial_data: Dict) -> Dict[str, Any]:
        """分析财务健康状况"""
        financials = financial_data.get("financials", {})
        return {
            "score": "良好" if financials.get("roe", 0) > 0.15 else "一般",
            "profitability": "强" if financials.get("profit_margins", 0) > 0.2 else "中等",
            "efficiency": "高" if financials.get("roe", 0) > 0.2 else "中等",
            "leverage": "适中" if financials.get("debt_to_equity", 1) < 1 else "偏高"
            }

    def _analyze_valuation(self, financial_data: Dict) -> Dict[str, Any]:
        """分析估值水平"""
        financials = financial_data.get("financials", {})
        pe = financials.get("pe_ratio", 0)

        if pe > 0:
            if pe < 15:
                valuation = "低估"
            elif pe < 25:
                valuation = "合理"
            else:
                valuation = "高估"
        else:
            valuation = "无法评估"

        return {
            "level": valuation,
            "pe_ratio": pe,
            "peg_ratio": financials.get("peg_ratio"),
            "price_to_book": financials.get("price_to_book")
            }

    def _analyze_growth(self, financial_data: Dict) -> Dict[str, Any]:
        """分析增长潜力"""
        financials = financial_data.get("financials", {})
        revenue_growth = financials.get("revenue_growth", 0)
        earnings_growth = financials.get("earnings_growth", 0)

        return {
            "revenue_growth": revenue_growth,
            "earnings_growth": earnings_growth,
            "growth_quality": "高" if revenue_growth > 0.15 and earnings_growth > 0.15 else "中等"
            }

    def _analyze_trend(self, technical_data: Dict) -> str:
        """分析价格趋势"""
        signals = technical_data.get("signals", {})
        ma_signal = signals.get("ma", "")

        if "多头排列" in ma_signal:
            return "上升趋势"
        elif "空头排列" in ma_signal:
            return "下降趋势"
        else:
            return "盘整"

    def _analyze_momentum(self, technical_data: Dict) -> Dict[str, Any]:
        """分析动量指标"""
        indicators = technical_data.get("indicators", {})
        return {
            "rsi": indicators.get("rsi", 50),
            "rsi_signal": technical_data.get("signals", {}).get("rsi", "中性"),
            "macd_signal": technical_data.get("signals", {}).get("macd", "中性")
            }

    def _identify_risks(self, stock_data: Dict) -> List[str]:
        """识别风险"""
        risks = []

        # 估值风险
        financials = stock_data.get("financials", {}).get("data", {}).get("financials", {})
        if financials.get("pe_ratio", 0) > 35:
            risks.append("估值偏高")

        # 技术风险
        technical = stock_data.get("technicals", {}).get("data", {}).get("indicators", {})
        if technical.get("rsi", 50) > 70:
            risks.append("技术指标超买")

        # 负债风险
        if financials.get("debt_to_equity", 0) > 2:
            risks.append("负债率偏高")

        return risks

    def _analyze_current_allocation(self, portfolio_data: Dict) -> Dict[str, Any]:
        """分析当前配置"""
        positions = portfolio_data.get("portfolio", {}).get("positions", [])
        total_value = sum(pos.get("value", 0) for pos in positions)

        allocation = {}
        for pos in positions:
            allocation[pos.get("symbol", "Unknown")] = pos.get("value", 0) / total_value if total_value > 0 else 0

        return {
            "allocation": allocation,
            "diversification": len(positions),
            "concentration_risk": max(allocation.values()) if allocation else 0
            }

    def _calculate_optimal_allocation(self, portfolio_data: Dict, risk_tolerance: str, market_data: Dict) -> Dict[
        str, Any]:
        """计算最优配置"""
        # 简化的资产配置建议
        if risk_tolerance == "conservative":
            return {
                "stocks": 0.4,
                "bonds": 0.4,
                "cash": 0.2
                }
        elif risk_tolerance == "aggressive":
            return {
                "stocks": 0.8,
                "bonds": 0.1,
                "cash": 0.1
                }
        else:  # moderate
            return {
                "stocks": 0.6,
                "bonds": 0.3,
                "cash": 0.1
                }

    def _generate_rebalancing_suggestions(self, portfolio_data: Dict, risk_tolerance: str) -> List[Dict]:
        """生成再平衡建议"""
        suggestions = []
        current_allocation = self._analyze_current_allocation(portfolio_data)

        # 简化的建议
        if current_allocation.get("concentration_risk", 0) > 0.3:
            suggestions.append(
                {
                    "action": "reduce",
                    "reason": "单一持仓占比过高",
                    "priority": "high"
                    }
                )

        return suggestions

    def _portfolio_risk_analysis(self, portfolio_data: Dict) -> Dict[str, Any]:
        """组合风险分析"""
        return {
            "volatility": "中等",
            "var_95": 0.05,  # 5% VaR
            "max_drawdown_risk": 0.15,
            "diversification_score": 0.7
            }

    def _analyze_market_trend(self, market_data: Dict) -> str:
        """分析市场趋势"""
        indices = market_data.get("indices", {})

        positive_count = sum(1 for idx in indices.values() if idx.get("change", 0) > 0)

        if positive_count >= 2:
            return "上涨"
        elif positive_count <= 1:
            return "下跌"
        else:
            return "混合"

    def _analyze_market_breadth(self, market_data: Dict) -> Dict[str, Any]:
        """分析市场广度"""
        breadth = market_data.get("market_breadth", {})
        return {
            "advance_decline_ratio": breadth.get("advance_decline", 1),
            "new_highs": breadth.get("new_highs", 0),
            "new_lows": breadth.get("new_lows", 0),
            "breadth_signal": "积极" if breadth.get("advance_decline", 1) > 1.5 else "消极"
            }

    def _analyze_sector_rotation(self, sectors: Dict) -> Dict[str, Any]:
        """分析板块轮动"""
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1].get("performance", 0), reverse=True)

        return {
            "leading_sectors": [s[0] for s in sorted_sectors[:3]],
            "lagging_sectors": [s[0] for s in sorted_sectors[-3:]],
            "rotation_signal": "科技股领涨" if sorted_sectors[0][0] == "technology" else "防御性板块领涨"
            }

    def _identify_opportunities(self, market_data: Dict) -> List[Dict]:
        """识别市场机会"""
        opportunities = []

        vix = market_data.get("vix", 20)
        if vix < 15:
            opportunities.append(
                {
                    "type": "low_volatility",
                    "description": "市场波动率低，适合买入看涨期权",
                    "confidence": "medium"
                    }
                )
        elif vix > 30:
            opportunities.append(
                {
                    "type": "high_volatility",
                    "description": "市场恐慌，可能存在抄底机会",
                    "confidence": "low"
                    }
                )

        return opportunities

    def _identify_market_risks(self, market_data: Dict) -> List[Dict]:
        """识别市场风险"""
        risks = []

        vix = market_data.get("vix", 20)
        if vix > 25:
            risks.append(
                {
                    "type": "high_volatility",
                    "description": "市场波动加剧",
                    "severity": "high"
                    }
                )

        breadth = market_data.get("market_breadth", {})
        if breadth.get("advance_decline", 1) < 0.5:
            risks.append(
                {
                    "type": "weak_breadth",
                    "description": "市场广度疲弱",
                    "severity": "medium"
                    }
                )

        return risks

    async def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict]:
        """生成具体建议"""
        recommendations = []

        # 根据分析结果生成建议
        if "rating" in analysis_result:
            rating = analysis_result["rating"]
            if rating in ["强烈买入", "买入"]:
                recommendations.append(
                    {
                        "action": "buy",
                        "reason": "综合评分较高",
                        "confidence": "high" if rating == "强烈买入" else "medium",
                        "risk_level": "medium"
                        }
                    )
            elif rating in ["卖出", "强烈卖出"]:
                recommendations.append(
                    {
                        "action": "sell",
                        "reason": "综合评分较低",
                        "confidence": "high" if rating == "强烈卖出" else "medium",
                        "risk_level": "low"
                        }
                    )

        # 风险提示
        if "risks" in analysis_result and len(analysis_result["risks"]) > 0:
            recommendations.append(
                {
                    "action": "monitor",
                    "reason": f"存在风险因素：{', '.join(analysis_result['risks'])}",
                    "confidence": "high",
                    "risk_level": "high"
                    }
                )

        return recommendations

    async def get_portfolio_status(self, user_id: str = "default") -> Dict[str, Any]:
        """获取组合状态"""
        portfolio_resource = self.mcp_server.resources[f"stock://portfolio/{user_id}"]
        portfolio_data = await portfolio_resource.read()

        return {
            "success": True,
            "portfolio": portfolio_data["data"],
            "summary": {
                "total_value": portfolio_data["data"]["portfolio"]["total_value"],
                "positions_count": len(portfolio_data["data"]["portfolio"]["positions"]),
                "cash_available": portfolio_data["data"]["portfolio"]["cash"]
                }
            }

    async def execute_trade(self, symbol: str, action: str, quantity: int, user_id: str = "default") -> Dict[str, Any]:
        """执行交易（模拟）"""
        portfolio_resource = self.mcp_server.resources[f"stock://portfolio/{user_id}"]

        # 获取当前价格
        price_data = await self.mcp_server.tools["get_stock_price"](symbol, "1d")
        if not price_data["success"]:
            return {"success": False, "error": "无法获取股票价格"}

        current_price = price_data["data"]["current_price"]

        # 更新组合
        if action == "buy":
            await portfolio_resource.add_position(
                {
                    "symbol": symbol,
                    "quantity": quantity,
                    "average_cost": current_price,
                    "current_price": current_price,
                    "value": current_price * quantity,
                    "purchase_date": datetime.now().isoformat()
                    }
                )
            portfolio_resource.portfolio["cash"] -= current_price * quantity

        elif action == "sell":
            # 简化处理，实际需要更复杂的逻辑
            positions = portfolio_resource.portfolio["positions"]
            for i, pos in enumerate(positions):
                if pos["symbol"] == symbol:
                    if pos["quantity"] >= quantity:
                        pos["quantity"] -= quantity
                        if pos["quantity"] == 0:
                            positions.pop(i)
                        portfolio_resource.portfolio["cash"] += current_price * quantity
                        break

        # 更新总值
        total_value = portfolio_resource.portfolio["cash"]
        for pos in portfolio_resource.portfolio["positions"]:
            total_value += pos["value"]
        portfolio_resource.portfolio["total_value"] = total_value

        return {
            "success": True,
            "trade": {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "price": current_price,
                "total_value": current_price * quantity,
                "timestamp": datetime.now().isoformat()
                }
            }

    # 辅助函数
    def format_analysis_report(analysis_result: Dict[str, Any]) -> str:
        """格式化分析报告"""
        report = []

        if "symbol" in analysis_result:
            report.append(f"## {analysis_result['symbol']} 股票分析报告\n")
            report.append(f"**综合评级**: {analysis_result.get('rating', 'N/A')}\n")

            # 基本面分析
            if "fundamental_analysis" in analysis_result:
                report.append("### 基本面分析")
                fa = analysis_result["fundamental_analysis"]
                report.append(f"- 财务健康: {fa.get('financial_health', {}).get('score', 'N/A')}")
                report.append(f"- 估值水平: {fa.get('valuation', {}).get('level', 'N/A')}")
                report.append(f"- 增长潜力: {fa.get('growth_potential', {}).get('growth_quality', 'N/A')}\n")

            # 技术面分析
            if "technical_analysis" in analysis_result:
                report.append("### 技术面分析")
                ta = analysis_result["technical_analysis"]
                report.append(f"- 价格趋势: {ta.get('trend', 'N/A')}")
                report.append(f"- RSI: {ta.get('momentum', {}).get('rsi', 'N/A'):.2f}")
                report.append(f"- 支撑位: {ta.get('support_resistance', {}).get('support', 'N/A')}")
                report.append(f"- 阻力位: {ta.get('support_resistance', {}).get('resistance', 'N/A')}\n")

            # 风险提示
            if "risks" in analysis_result and analysis_result["risks"]:
                report.append("### 风险提示")
                for risk in analysis_result["risks"]:
                    report.append(f"- {risk}")

        return "\n".join(report)
