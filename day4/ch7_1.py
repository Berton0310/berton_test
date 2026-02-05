import random
import time
import json
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
import os

# ================= 配置與快取函式 =================
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="EMPTY",
    model="google/gemma-3-27b-it",
    temperature=0.7
)

CACHE_FILE = "translation_cache.json"

def load_cache():
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_cache(original: str, translated: str):
    data = load_cache()
    data[original] = translated
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ================= 1. 定義狀態 =================
class State(TypedDict):
    original_text: str
    translated_text: str
    critique: str
    attempts: int
    is_cache_hit: bool # 標記是否命中快取

def check_cache_node(state: State):
    """檢查快取節點"""
    print("\n--- 檢查快取 (Check Cache) ---")
    data = load_cache()
    original = state["original_text"]

    if original in data:
        print("✅ 命中快取！直接回傳結果。")
        return {
            "translated_text": data[original],
            "is_cache_hit": True
        }
    else:
        print("❌ 未命中快取，準備開始翻譯流程...")
        return {"is_cache_hit": False}

def translator_node(state: State):
    """翻譯節點"""
    print(f"\n--- 翻譯嘗試 (第 {state['attempts'] + 1} 次) ---")
    prompt = f"你是一名翻譯員，請將以下中文翻譯成英文，不須任何解釋：'{state['original_text']}'"
    if state['critique']:
        prompt += f"\n\n上一輪的審查意見是：{state['critique']}。請根據意見修正翻譯。"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"translated_text": response.content, "attempts": state['attempts'] + 1}

def reflector_node(state: State):
    """審查節點"""
    print("--- 審查中 (Reflection) ---")
    prompt = f"原文：{state['original_text']}\n翻譯：{state['translated_text']}\n請檢查翻譯是否準確，若準確請回覆 PASS，否則請給出修正建議。"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"critique": response.content}

# ================= 3. 定義路由 (Routers) =================

def cache_router(state: State) -> Literal["end", "translator"]:
    """[新增] 快取路由：有快取就結束，沒快取就去翻譯"""
    if state["is_cache_hit"]:
        return "end"
    return "translator"

def critique_router(state: State) -> Literal["translator", "end"]:
    """審查路由"""
    if "PASS" in state["critique"].upper():
        print("--- 審查通過！ ---")
        return "end"
    elif state["attempts"] >= 3:
        print("--- 達到最大重試次數 ---")
        return "end"
    else:
        print(f"--- 退回重寫：{state['critique']} ---")
        return "translator"

workflow = StateGraph(State)

# 加入節點
workflow.add_node("check_cache", check_cache_node) # 新節點
workflow.add_node("translator", translator_node)
workflow.add_node("reflector", reflector_node)

# 一律先走 check_cache
workflow.set_entry_point("check_cache")

# 設定快取後的路徑 (Cache Hit -> END; Cache Miss -> Translator)
workflow.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "end": END,
        "translator": "translator"
    }
)

# 正常的翻譯迴圈路徑
workflow.add_edge("translator", "reflector")
workflow.add_conditional_edges(
    "reflector",
    critique_router,
    {"translator": "translator", "end": END}
)

app = workflow.compile()
print(app.get_graph().draw_ascii())

if __name__ == "__main__":
    print(f"快取檔案: {CACHE_FILE}")

    while True:
        user_input = input("\n請輸入要翻譯的中文 (exit/q 離開): ")
        if user_input.lower() in ["exit", "q"]: break

        inputs = {
            "original_text": user_input,
            "attempts": 0,
            "critique": "",
            "is_cache_hit": False,
            "translated_text": "" # 初始為空
        }

        # 執行 Graph
        result = app.invoke(inputs)

        # 如果不是從快取來的（代表是新算出來的），就寫入快取
        if not result["is_cache_hit"]:
            save_cache(result["original_text"], result["translated_text"])
            print("(已將新翻譯寫入快取)")

        print("\n=========== 最終結果 ===========")
        print(f"原文: {result['original_text']}")
        print(f"翻譯: {result['translated_text']}")
        print(f"來源: {'快取 (Cache)' if result['is_cache_hit'] else '生成 (LLM)'}")
