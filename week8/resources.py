# resources.py
from datetime import datetime
from typing import Dict, Any

from mcp_stock_agent import MCPServer


class StockResource:
    """股票数据资源"""

    def __init__(self, uri: str):
        self.uri = uri
        self.data = {}

    async def read(self) -> Dict[str, Any]:
        """读取资源数据"""
        return {
            "uri": self.uri,
            "mime_type": "application/json",
            "data": self.data
            }

    async def write(self, data: Dict[str, Any]):
        """写入资源数据"""
        self.data = data


class MarketDataResource(StockResource):
    """市场数据资源"""

    async def read(self) -> Dict[str, Any]:
        """读取市场数据"""
        return {
            "uri": self.uri,
            "mime_type": "application/json",
            "data": {
                "indices": {
                    "SP500": {"value": 4500.5, "change": 0.5},
                    "NASDAQ": {"value": 14000.2, "change": 0.8},
                    "DJI": {"value": 35000.1, "change": 0.3}
                    },
                "sectors": {
                    "technology": {"performance": 1.2},
                    "healthcare": {"performance": -0.3},
                    "finance": {"performance": 0.5}
                    },
                "market_breadth": {
                    "advance_decline": 1.5,
                    "new_highs": 120,
                    "new_lows": 45
                    },
                "vix": 18.5,
                "updated_at": datetime.now().isoformat()
                }
            }


class PortfolioResource(StockResource):
    """投资组合资源"""

    def __init__(self, uri: str, user_id: str):
        super().__init__(uri)
        self.user_id = user_id
        self.portfolio = {
            "positions": [],
            "cash": 10000.0,
            "total_value": 10000.0
            }

    async def read(self) -> Dict[str, Any]:
        """读取投资组合"""
        return {
            "uri": self.uri,
            "mime_type": "application/json",
            "data": {
                "user_id": self.user_id,
                "portfolio": self.portfolio,
                "performance": await self._calculate_performance(),
                "updated_at": datetime.now().isoformat()
                }
            }

    async def _calculate_performance(self) -> Dict[str, Any]:
        """计算组合表现"""
        # 简化的性能计算
        return {
            "total_return": 0.0,
            "daily_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0
            }

    async def add_position(self, position: Dict[str, Any]):
        """添加持仓"""
        self.portfolio["positions"].append(position)

    async def update_position(self, symbol: str, updates: Dict[str, Any]):
        """更新持仓"""
        for pos in self.portfolio["positions"]:
            if pos["symbol"] == symbol:
                pos.update(updates)
                break


class WatchlistResource(StockResource):
    """自选股资源"""

    def __init__(self, uri: str, user_id: str):
        super().__init__(uri)
        self.user_id = user_id
        self.watchlist = []

    async def read(self) -> Dict[str, Any]:
        """读取自选股列表"""
        return {
            "uri": self.uri,
            "mime_type": "application/json",
            "data": {
                "user_id": self.user_id,
                "watchlist": self.watchlist,
                "count": len(self.watchlist),
                "updated_at": datetime.now().isoformat()
                }
            }

    async def add_symbol(self, symbol: str, metadata: Dict[str, Any] = None):
        """添加股票到自选"""
        self.watchlist.append(
            {
                "symbol": symbol,
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {}
                }
            )

    async def remove_symbol(self, symbol: str):
        """从自选中移除股票"""
        self.watchlist = [w for w in self.watchlist if w["symbol"] != symbol]


class StrategyResource(StockResource):
    """交易策略资源"""

    def __init__(self, uri: str):
        super().__init__(uri)
        self.strategies = {}

    async def read(self) -> Dict[str, Any]:
        """读取策略列表"""
        return {
            "uri": self.uri,
            "mime_type": "application/json",
            "data": {
                "strategies": self.strategies,
                "count": len(self.strategies),
                "updated_at": datetime.now().isoformat()
                }
            }

    async def add_strategy(self, name: str, config: Dict[str, Any]):
        """添加策略"""
        self.strategies[name] = {
            "name": name,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "status": "inactive"
            }

    async def update_strategy_status(self, name: str, status: str):
        """更新策略状态"""
        if name in self.strategies:
            self.strategies[name]["status"] = status


# 资源注册函数
def register_resources(mcp_server: MCPServer, user_id: str = "default"):
    """注册所有资源到 MCP 服务器"""

    mcp_server.resources = {
        "stock://market/data": MarketDataResource("stock://market/data"),
        f"stock://portfolio/{user_id}": PortfolioResource(f"stock://portfolio/{user_id}", user_id),
        f"stock://watchlist/{user_id}": WatchlistResource(f"stock://watchlist/{user_id}", user_id),
        "stock://strategies/list": StrategyResource("stock://strategies/list")
        }
