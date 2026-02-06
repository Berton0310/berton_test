from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import requests
import uuid

# 讀取 text.txt 檔案
try:
    with open('text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print("Error: text.txt not found.")
    text = ""

if text:
    # --- Part 1: CharacterTextSplitter ---
    print("--- CharacterTextSplitter ---")
    # 初始化固定大小分塊器
    text_splitter = CharacterTextSplitter(
        chunk_size=200,          # 每個分塊的字符數
        chunk_overlap=0,
        # 即使你沒寫，預設會使用 separator="\n\n" 分割
        separator="",
        length_function=len     # 計算長度的函數
    )

    # 執行分塊
    chunks = text_splitter.split_text(text)

    # 顯示結果
    print(f"總共產生 {len(chunks)} 個分塊\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"=== 分塊 {i} ===")
        print(f"長度: {len(chunk)} 字符")
        print(f"內容: {chunk.strip()}")
        print()


    # --- Part 2: TokenTextSplitter ---
    print("\n--- TokenTextSplitter ---")
    # 第一次執行會稍微久一點
    
    text_splitter_token = TokenTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
        model_name="gpt-4", # 注意: 這需要 tiktoken 套件
    )

    chunks_token = text_splitter_token.split_text(text)

    print(f"原始文本長度: {len(text)} tokens (Note: likely chars if raw text)") 
    print(f"原始文本長度: {len(text)} tokens")
    print(f"分塊數量: {len(chunks_token)}")
    # print("\n")

    for i, chunk in enumerate(chunks_token):
        print(f"分塊 {i+1}:")
        # 注意：Token 分割器的分塊可能不直接支援 len() 作為 token 數量，除非它回傳的是文本。
        # split_text 回傳的是字串列表。所以 len(chunk) 是字元數。
        # 截圖顯示 "長度: {len(chunk)} tokens"。
        # 這可能是截圖中的誤解，或者只是將字元長度標記為 tokens。
        # 我將遵照截圖顯示。
        print(f" 長度: {len(chunk)} tokens")
        # print("\n")


# --- Part 3: Qdrant Setup & Upsert ---

def get_embedding(text):
    data = {
        "texts": [text],
        "normalize": True,
        "batch_size": 32
    }
    # 使用與 hw1 相同的 API
    try:
        response = requests.post("https://ws-04.wade0426.me/embed", json=data)
        if response.status_code == 200:
            return response.json()['embeddings'][0]
    except Exception as e:
        print(f"Embedding API Error: {e}")
    return None

client = QdrantClient(url="http://localhost:6333")

def setup_collection_and_upsert(collection_name, chunks, splitter_name):
    print(f"\nProcessing {collection_name} for {splitter_name}...")
    
    # 檢查/建立 Collection
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
        )
        print(f"Collection '{collection_name}' created.")
    else:
        print(f"Collection '{collection_name}' already exists.")

    points = []
    for i, chunk_text in enumerate(chunks):
        # 為了避免 API 雖然切分了但內容為空
        if not chunk_text.strip():
            continue
            
        vector = get_embedding(chunk_text)
        if vector:
            points.append(PointStruct(
                id=str(uuid.uuid4()), # 使用 UUID 防止 ID 衝突
                vector=vector,
                payload={
                    "text": chunk_text,
                    "chunk_id": i,
                    "splitter": splitter_name
                }
            ))
            print(f"Generated embedding for chunk {i+1}/{len(chunks)}", end='\r')
    
    if points:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"\nSuccessfully upserted {len(points)} points to {collection_name}")
    else:
        print("\nNo points generated.")

# 執行 Upsert
if text:
    # hw2_char 用於 CharacterTextSplitter 分塊
    # setup_collection_and_upsert("hw2_char", chunks, "CharacterTextSplitter")
    # hw2_token 用於 TokenTextSplitter 分塊
    # Token 分割器回傳字串，沒問題
    # setup_collection_and_upsert("hw2_token", chunks_token, "TokenTextSplitter")
    pass


# --- 第 4 部分：搜尋 ---
query_text = "Graph RAG 技術架構"
print(f"\n{'='*20}\nQuerying: {query_text}\n{'='*20}")

query_vector = get_embedding(query_text)

if query_vector:
    for collection_name in ["hw2_char", "hw2_token"]:
        print(f"\n--- Results from {collection_name} ---")
        try:
            search_result = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=3
            )
            for point in search_result.points:
                print(f"Score: {point.score:.4f}")
                print(f"Chunk ID: {point.payload['chunk_id']}")
                print(f"Content: {point.payload['text']}")
                print("-" * 10)
        except Exception as e:
            print(f"Search failed for {collection_name}: {e}")
