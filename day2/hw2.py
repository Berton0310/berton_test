from langchain_openai import ChatOpenAI
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# 1. 設定模型 (LLM)
llm = ChatOpenAI(
    model="Llama-3.3-70B-Instruct-NVFP4",
    base_url="https://ws-03.wade0426.me/v1",
    api_key="EMPTY", 
    temperature=0,  
    max_tokens=100
)

# 2. 建立 Prompt Templates (提示詞模板)
# 風格 1: 感性/情緒化
prompt_sentimental = ChatPromptTemplate.from_messages([
    ("system", "你是一位充滿情感、語氣溫暖且富有感染力的社群小編。"),
    ("user", "請為主題「{topic}」寫一句話感性的貼文，著重於個人感受與情感連結，包含標籤。")
])

# 風格 2: 理性/專業
prompt_rational = ChatPromptTemplate.from_messages([
    ("system", "你是一位專業、客觀且邏輯嚴謹的分析師。"),
    ("user", "請為主題「{topic}」寫一句理性的分析文，著重於事實、數據與邏輯推演，包含標籤")
])

# 3. 建立鏈 (Chain)
chain_sentimental = prompt_sentimental | llm | StrOutputParser()
chain_rational = prompt_rational | llm | StrOutputParser()

# 4. 平行處理 (Parallel Execution)
map_chain = RunnableParallel(
    sentimental=chain_sentimental,
    rational=chain_rational
)

# 5. 執行調用
if __name__ == "__main__":
    try:
        topic = input("請輸入主題: ")
        print(f"\n正在為主題「{topic}」生成貼文...\n")

        # --- Streaming (串流執行) ---
        print("===" * 10)
        print(" [Stream 模式] ")
        print("===" * 10)

        print("(正在串流輸出... 格式為 raw chunks)")
        for chunk in map_chain.stream({"topic": topic}):
            # chunk 是一個字典，包含部分生成的內容
            try:
                print(chunk,end='',flush=True)
            except UnicodeEncodeError:
                print(chunk.encode('utf-8','replace').decode('utf-8'),end='',flush=True)
        # --- Batch Invoke (批次執行) ---
        print("===" * 10)
        print(" [Batch Invoke 模式] ")
        print("===" * 10)
        
        start_time = time.time()
        result = map_chain.invoke({"topic": topic})
        end_time = time.time()
        print(f"耗時: {end_time - start_time:.4f} 秒")
        
        print("\n📝 [感性風格]:")
        print(result["sentimental"])
        print("\n📊 [理性風格]:")
        print(result["rational"])
        print("\n")

        

    except Exception as e:
        print(f"發生錯誤: {e}")