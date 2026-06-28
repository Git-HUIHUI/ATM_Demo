"""
Chainlit 前端 - 空管智能助手 (Chainlit 2.x)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chainlit as cl
from langchain_core.messages import HumanMessage, AIMessage

from core.agent_graph import create_agent


@cl.on_chat_start
async def start():
    agent = create_agent()
    cl.user_session.set("agent", agent)
    cl.user_session.set("history", [])
    await cl.Message(
        content="**空管智能助手** 已就绪。\n\n试试：\n- 查询成都双流今天有多少航班\n- ZUUU最近流量怎么样\n- 什么是A类空域"
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    agent = cl.user_session.get("agent")
    history = cl.user_session.get("history", [])
    config = {"configurable": {"thread_id": cl.user_session.get("id")}}

    history.append(HumanMessage(content=msg.content))

    response = cl.Message(content="")
    async for event in agent.astream_events(
        {"messages": history}, config, version="v2",
    ):
        kind = event.get("event")
        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                await response.stream_token(token)
        elif kind == "on_tool_start":
            await response.stream_token(f"\n> 🔧 调用工具: **{event['name']}**\n")
        elif kind == "on_tool_end":
            output = event["data"].get("output", "")
            if isinstance(output, str) and output:
                lines = output.strip().split("\n")
                summary_lines = [l for l in lines if l.startswith("##") or l.startswith("- ")]
                preview = "\n> ".join(summary_lines[:10])
                await response.stream_token(f"> {preview}\n")
                if len(summary_lines) > 10:
                    await response.stream_token("> *(完整数据见上方工具输出)*\n")
    await response.send()
    history.append(AIMessage(content=response.content))
    cl.user_session.set("history", history)
