from fastmcp import Client

import asyncio

async def main():
    async with Client("weather_server.py") as mcp_client:

        # Get available tools
        tools = await mcp_client.list_tools()
        print(f"Available tools: {tools}")

        # Call a tool
        result = await mcp_client.call_tool("get_weather", {"city": "Berlin"})
        print(f"Weather in Berlin: {result}")

        # Call another tool
        result = await mcp_client.call_tool("set_weather", {"city": "Berlin", "temp": 20.5})
        print(f"Set weather result: {result}")

if __name__ == "__main__":
    test = asyncio.run(main())