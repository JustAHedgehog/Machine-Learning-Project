import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import chromadb
import uuid
import os
import shutil

# 讀取CSV數據
df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()

# 定義輸入特徵 (查詢用)
input_cols = ['穿著本套衣服時的溫度(°C)', '穿著本套衣服時的體感溫度 (°C)', 
              '穿著本套衣服時的相對濕度(%)', '性別', '本套衣服今日主要活動範圍']
# 定義輸出結果 (推薦結果)
output_cols = ['上衣', '外套', '褲子 / 裙子', '為了因應天氣所配戴的配件']

# 數據預處理流水線 (數值和類別欄位分別處理)
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, ['穿著本套衣服時的溫度(°C)', 
                                                  '穿著本套衣服時的體感溫度 (°C)', 
                                                  '穿著本套衣服時的相對濕度(%)']),
                  ('cat', categorical_transformer, ['性別', '本套衣服今日主要活動範圍'])]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# 處理輸入特徵向量
X_inputs = pipeline.fit_transform(df[input_cols])

# 如果資料夾存在，清空
folder_path = "my_database"
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    shutil.rmtree(folder_path)

# 初始化 Chroma 資料庫
persist_directory = './my_database'
client = chromadb.PersistentClient(path=persist_directory)
collection = client.create_collection("outfit_recommendations")

# 輸出文檔 (合併推薦結果)
documents = df[output_cols].astype(str).agg(' '.join, axis=1).tolist()

# 元數據 (存儲輸入條件以便查詢)
metadatas = df[input_cols].to_dict(orient='records')

# 唯一 ID
ids = [str(uuid.uuid4()) for _ in range(len(df))]

# 將數據插入 Chroma 資料庫
collection.add(
    documents=documents,
    metadatas=metadatas,
    embeddings=X_inputs.tolist(),  # 使用處理後的特徵向量
    ids=ids
)

print(f"插入了 {len(df)} 條資料進入 Chroma 資料庫")

# 查詢測試：查詢條件
query_data = {
    '穿著本套衣服時的溫度(°C)': [22],
    '穿著本套衣服時的體感溫度 (°C)': [21],
    '穿著本套衣服時的相對濕度(%)': [60],
    '性別': ['男'],
    '本套衣服今日主要活動範圍': ['室內']
}
query_df = pd.DataFrame(query_data)

# 處理查詢數據的向量化
query_inputs = pipeline.transform(query_df)
query_embeddings = query_inputs.flatten()

# 查詢 Chroma 資料庫 (找最相近的 3 筆結果)
query_results = collection.query(query_embeddings=query_embeddings, n_results=3)

# 顯示結果
for idx, (document, metadata, distance) in enumerate(
    zip(query_results["documents"], query_results["metadatas"], query_results["distances"])
):
    print(f"Result {idx + 1}:")
    print(f"Recommended Outfit: {document}")  # 推薦穿著
    print(f"Query Metadata: {metadata}")  # 原始條件
    print(f"Similarity Distance: {distance}")  # 相似度
    print("-----")