# main.py
import asyncio
import json
import logging
from typing import Dict, Any

from mcp_stock_agent import MCPServer
from prompts import STOCK_ADVISOR_PROMPTS
from resources import register_resources
from stock_advisor_agent import StockAdvisorAgent
from tools import register_tools

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


class StockAdvisorSystem:
    """股票顾问系统"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mcp_server = MCPServer()
        self.agent = None

    async def initialize(self):
        """初始化系统"""
        logger.info("初始化股票顾问系统...")

        # 注册 Prompts
        self.mcp_server.prompts = STOCK_ADVISOR_PROMPTS
        logger.info(f"注册了 {len(self.mcp_server.prompts)} 个 prompts")

        # 注册 Tools
        api_keys = self.config.get("api_keys", {})
        register_tools(self.mcp_server, api_keys)
        logger.info(f"注册了 {len(self.mcp_server.tools)} 个 tools")

        # 注册 Resources
        register_resources(self.mcp_server)
        logger.info(f"注册了 {len(self.mcp_server.resources)} 个 resources")

        # 创建 Agent
        self.agent = StockAdvisorAgent(self.mcp_server)
        logger.info("股票顾问 Agent 创建成功")

    async def run_interactive(self):
        """运行交互式会话"""
        print("\n🤖 欢迎使用 AI 股票投资顾问")
        print("=" * 50)
        print("我可以帮您：")
        print("1. 分析个股（输入如：分析AAPL）")
        print("2. 优化投资组合")
        print("3. 市场趋势分析")
        print("4. 股票筛选推荐")
        print("5. 查看持仓状态（输入：我的组合）")
        print("\n输入 'exit' 退出，'help' 查看帮助")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n💬 请输入您的问题: ").strip()

                if user_input.lower() == 'exit':
                    print("👋 感谢使用，再见！")
                    break

                elif user_input.lower() == 'help':
                    await self.show_help()
                    continue

                elif "我的组合" in user_input or "持仓" in user_input:
                    result = await self.agent.get_portfolio_status()
                    self.display_portfolio(result)
                    continue

                # 处理用户请求
                print("\n🔄 正在分析...")
                result = await self.agent.process_request(user_input)

                if result["success"]:
                    self.display_result(result)
                else:
                    print(f"\n❌ 错误: {result.get('error', '未知错误')}")

            except KeyboardInterrupt:
                print("\n\n👋 感谢使用，再见！")
                break
            except Exception as e:
                logger.error(f"处理请求时出错: {e}", exc_info=True)
                print(f"\n❌ 系统错误: {str(e)}")

    async def show_help(self):
        """显示帮助信息"""
        help_text = """
📋 使用帮助：

1. 个股分析
   - 示例：分析 AAPL
   - 示例：TSLA 的技术分析
   - 示例：评估 MSFT 的投资价值

2. 投资组合
   - 示例：优化我的投资组合
   - 示例：我的风险承受能力是保守的，如何调整组合
   - 示例：查看我的持仓

3. 市场分析
   - 示例：当前市场趋势如何
   - 示例：哪些板块值得关注
   - 示例：市场风险评估

4. 股票筛选
   - 示例：推荐一些低市盈率的股票
   - 示例：找一些高增长的科技股
   - 示例：有哪些分红高的股票

5. 交易执行（模拟）
   - 示例：买入 100 股 AAPL
   - 示例：卖出 50 股 TSLA

提示：
- 股票代码请使用大写字母（如 AAPL, MSFT）
- 可以同时询问多只股票
- 所有建议仅供参考，请谨慎投资
        """
        print(help_text)

    def display_result(self, result: Dict[str, Any]):
        """显示分析结果"""
        print("\n" + "=" * 60)
        print("📊 分析结果")
        print("=" * 60)

        analysis = result.get("analysis", {})

        # 显示格式化的分析报告
        if analysis:
            report = self.agent.format_analysis_report(analysis)
            print(report)

        # 显示建议
        recommendations = result.get("recommendations", [])
        if recommendations:
            print("\n💡 投资建议：")
            for i, rec in enumerate(recommendations, 1):
                action_emoji = {
                    "buy": "📈",
                    "sell": "📉",
                    "hold": "⏸️",
                    "monitor": "👁️"
                    }.get(rec.get("action", ""), "💭")

                print(f"{i}. {action_emoji} {rec.get('reason', '')}")
                print(f"   置信度: {rec.get('confidence', 'N/A')}, 风险等级: {rec.get('risk_level', 'N/A')}")

        # 显示详细分析数据（如果需要）
        if "intent" in result and result["intent"].get("type") == "portfolio_optimization":
            self._display_portfolio_analysis(analysis)
        elif "intent" in result and result["intent"].get("type") == "market_insight":
            self._display_market_analysis(analysis)

        print("\n" + "=" * 60)

    def _display_portfolio_analysis(self, analysis: Dict[str, Any]):
        """显示投资组合分析"""
        if "current_allocation" in analysis:
            print("\n📊 当前资产配置：")
            allocation = analysis["current_allocation"].get("allocation", {})
            for asset, weight in allocation.items():
                print(f"  - {asset}: {weight:.1%}")

        if "optimal_allocation" in analysis:
            print("\n🎯 建议资产配置：")
            optimal = analysis["optimal_allocation"]
            for asset_type, weight in optimal.items():
                print(f"  - {asset_type}: {weight:.1%}")

        if "risk_analysis" in analysis:
            print("\n⚠️ 风险分析：")
            risk = analysis["risk_analysis"]
            print(f"  - 波动率: {risk.get('volatility', 'N/A')}")
            print(f"  - 95% VaR: {risk.get('var_95', 0):.1%}")
            print(f"  - 最大回撤风险: {risk.get('max_drawdown_risk', 0):.1%}")

    def _display_market_analysis(self, analysis: Dict[str, Any]):
        """显示市场分析"""
        if "market_overview" in analysis:
            overview = analysis["market_overview"]
            print("\n📈 市场概览：")
            print(f"  - 趋势: {overview.get('trend', 'N/A')}")
            print(f"  - 市场广度: {overview.get('breadth', {}).get('breadth_signal', 'N/A')}")
            print(f"  - VIX 波动率: {overview.get('volatility', 'N/A')}")

        if "sector_rotation" in analysis:
            rotation = analysis["sector_rotation"]
            print("\n🔄 板块轮动：")
            print(f"  - 领涨板块: {', '.join(rotation.get('leading_sectors', []))}")
            print(f"  - 落后板块: {', '.join(rotation.get('lagging_sectors', []))}")
            print(f"  - 轮动信号: {rotation.get('rotation_signal', 'N/A')}")

        if "opportunities" in analysis:
            opportunities = analysis["opportunities"]
            if opportunities:
                print("\n💡 市场机会：")
                for opp in opportunities:
                    print(f"  - {opp.get('description', '')}")

    def display_portfolio(self, result: Dict[str, Any]):
        """显示投资组合状态"""
        if not result["success"]:
            print(f"❌ 错误: {result.get('error', '无法获取组合信息')}")
            return

        print("\n" + "=" * 60)
        print("💼 我的投资组合")
        print("=" * 60)

        summary = result.get("summary", {})
        print(f"\n📊 组合概览：")
        print(f"  - 总市值: ${summary.get('total_value', 0):,.2f}")
        print(f"  - 持仓数量: {summary.get('positions_count', 0)}")
        print(f"  - 可用现金: ${summary.get('cash_available', 0):,.2f}")

        portfolio = result.get("portfolio", {}).get("portfolio", {})
        positions = portfolio.get("positions", [])

        if positions:
            print("\n📈 持仓明细：")
            print(f"{'股票':<6} {'数量':>8} {'成本':>10} {'现价':>10} {'市值':>12} {'盈亏':>10} {'盈亏%':>8}")
            print("-" * 76)

            for pos in positions:
                symbol = pos.get("symbol", "N/A")
                quantity = pos.get("quantity", 0)
                avg_cost = pos.get("average_cost", 0)
                current_price = pos.get("current_price", 0)
                value = pos.get("value", 0)
                pnl = (current_price - avg_cost) * quantity
                pnl_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0

                print(
                    f"{symbol:<6} {quantity:>8} ${avg_cost:>9.2f} ${current_price:>9.2f} "
                    f"${value:>11,.2f} ${pnl:>9,.2f} {pnl_pct:>7.1f}%"
                    )

        print("\n" + "=" * 60)

    async def run_demo(self):
        """运行演示模式"""
        print("\n🎯 运行演示模式...")

        # 演示场景列表
        demo_scenarios = [
            "分析 AAPL 的投资价值",
            "我想优化投资组合，风险承受能力中等",
            "当前市场趋势如何？有什么投资机会？",
            "推荐一些低市盈率高增长的股票"
            ]

        for i, scenario in enumerate(demo_scenarios, 1):
            print(f"\n\n{'=' * 60}")
            print(f"场景 {i}: {scenario}")
            print('=' * 60)

            result = await self.agent.process_request(scenario)
            if result["success"]:
                self.display_result(result)
            else:
                print(f"❌ 错误: {result.get('error', '未知错误')}")

            # 模拟用户阅读时间
            await asyncio.sleep(2)

    async def run_batch(self, requests_file: str):
        """批量处理请求"""
        try:
            with open(requests_file, 'r', encoding='utf-8') as f:
                requests = json.load(f)

            results = []
            for req in requests:
                user_input = req.get("query", "")
                print(f"\n处理请求: {user_input}")

                result = await self.agent.process_request(user_input)
                results.append(
                    {
                        "query": user_input,
                        "result": result
                        }
                    )

            # 保存结果
            output_file = requests_file.replace('.json', '_results.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"\n✅ 批量处理完成，结果保存在: {output_file}")

        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            print(f"❌ 批量处理失败: {e}")


async def main():
    """主函数"""
    # 系统配置
    config = {
        "api_keys": {
            # 在实际使用中，应该从环境变量或配置文件读取
            "openai": "your-openai-api-key",
            "finnhub": "your-finnhub-api-key",
            "alpha_vantage": "your-alpha-vantage-key"
            },
        "model_config": {
            "temperature": 0.7,
            "max_tokens": 2000
            }
        }

    # 创建并初始化系统
    system = StockAdvisorSystem(config)
    await system.initialize()

    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='AI 股票投资顾问')
    parser.add_argument(
        '--mode', choices=['interactive', 'demo', 'batch'],
        default='interactive', help='运行模式'
        )
    parser.add_argument('--file', type=str, help='批量模式的输入文件')

    args = parser.parse_args()

    # 根据模式运行
    try:
        if args.mode == 'interactive':
            await system.run_interactive()
        elif args.mode == 'demo':
            await system.run_demo()
        elif args.mode == 'batch':
            if not args.file:
                print("❌ 批量模式需要指定输入文件: --file <filename>")
            else:
                await system.run_batch(args.file)
    except Exception as e:
        logger.error(f"系统运行错误: {e}", exc_info=True)
        print(f"\n❌ 系统错误: {e}")


if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
