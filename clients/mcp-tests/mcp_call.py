"""Call an llm-mcp instance: list tools, then one call per entry.

  python mcp/mcp_call.py URL TOKEN '[["rerank", {"query": "...", "documents": ["..."]}]]'
"""
import asyncio, json, sys
import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL, TOKEN = sys.argv[1], sys.argv[2]
CALLS = json.loads(sys.argv[3])

async def main():
    async with httpx.AsyncClient(headers={"Authorization": f"Bearer {TOKEN}"}, timeout=1800) as hc:
        async with streamable_http_client(URL, http_client=hc) as (r, w, *_):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = await s.list_tools()
                print("TOOLS", [t.name for t in tools.tools])
                for name, args in CALLS:
                    res = await s.call_tool(name, args)
                    for c in res.content:
                        if c.type == "text":
                            print(f"[{name}] TEXT", c.text[:700])
                        else:
                            print(f"[{name}] {c.type} {getattr(c,'mimeType','')} {len(getattr(c,'data','') or '')} b64 chars")
asyncio.run(main())
