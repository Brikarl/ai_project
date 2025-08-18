# mcp_stock_agent.py
from typing import Dict, Any


# MCP 核心组件
class MCPServer:
    """MCP 服务器实现"""

    def __init__(self):
        self.tools = {}
        self.resources = {}
        self.prompts = {}

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 MCP 请求"""
        method = request.get("method")
        params = request.get("params", {})

        if method == "tools/list":
            return {"tools": list(self.tools.keys())}
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_params = params.get("arguments", {})
            return await self.tools[tool_name](**tool_params)
        elif method == "resources/list":
            return {"resources": list(self.resources.keys())}
        elif method == "resources/read":
            resource_uri = params.get("uri")
            return await self.resources[resource_uri].read()
        elif method == "prompts/list":
            return {"prompts": list(self.prompts.keys())}
        elif method == "prompts/get":
            prompt_name = params.get("name")
            return self.prompts[prompt_name]
