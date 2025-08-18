# prompts.py
STOCK_ADVISOR_PROMPTS = {
    "stock_analysis": {
        "name": "stock_analysis",
        "description": "分析股票的基本面和技术面",
        "arguments": [
            {
                "name": "symbol",
                "description": "股票代码",
                "required": True
                },
            {
                "name": "analysis_type",
                "description": "分析类型：fundamental/technical/both",
                "required": False
                }
            ],
        "template": """
你是一位专业的股票分析师，请对股票 {symbol} 进行{analysis_type}分析。

分析要求：
1. 基本面分析：
   - 财务状况（营收、利润、资产负债率）
   - 行业地位和竞争优势
   - 管理层质量
   - 未来增长潜力

2. 技术面分析：
   - 价格趋势和关键支撑/阻力位
   - 成交量分析
   - 技术指标（MA、RSI、MACD等）
   - 形态分析

3. 风险评估：
   - 市场风险
   - 行业风险
   - 公司特有风险

请基于以下数据进行分析：
{stock_data}

输出格式：
- 总体评级（强烈买入/买入/持有/卖出/强烈卖出）
- 详细分析报告
- 操作建议
- 风险提示
"""
        },

    "portfolio_optimization": {
        "name": "portfolio_optimization",
        "description": "优化投资组合配置",
        "arguments": [
            {
                "name": "current_portfolio",
                "description": "当前持仓",
                "required": True
                },
            {
                "name": "risk_tolerance",
                "description": "风险承受能力：conservative/moderate/aggressive",
                "required": True
                },
            {
                "name": "investment_goal",
                "description": "投资目标",
                "required": False
                }
            ],
        "template": """
作为专业的投资组合管理师，请为客户优化投资组合。

当前投资组合：
{current_portfolio}

风险承受能力：{risk_tolerance}
投资目标：{investment_goal}

请提供：
1. 当前组合分析
   - 资产配置比例
   - 风险收益特征
   - 相关性分析

2. 优化建议
   - 调整方案
   - 预期收益
   - 风险控制

3. 执行计划
   - 具体操作步骤
   - 时间安排
   - 注意事项
"""
        },

    "market_insight": {
        "name": "market_insight",
        "description": "市场洞察和趋势分析",
        "template": """
请提供当前市场的深度分析：

1. 宏观经济环境
   - GDP、通胀、利率走势
   - 货币政策影响
   - 地缘政治因素

2. 行业板块分析
   - 热门板块轮动
   - 行业景气度
   - 资金流向

3. 市场情绪
   - 投资者信心指数
   - 恐慌贪婪指数
   - 期权市场信号

4. 投资机会
   - 短期机会
   - 中长期布局
   - 风险提示

基于数据：{market_data}
"""
        }
    }
