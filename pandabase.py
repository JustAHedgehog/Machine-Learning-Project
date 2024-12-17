import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import chromadb
import uuid
import os
import shutil

# 讀取CSV
df = pd.read_csv('data.csv')
df.columns = df.columns.str.strip()
#print(df.columns) #check columns.title
folder_path = "my_database"
# 選擇數值欄位
numerical_cols = ['穿著本套衣服時的溫度(°C)', '穿著本套衣服時的體感溫度 (°C)', '穿著本套衣服時的相對濕度(%)']

# 選擇類別欄位
categorical_cols = ['性別', '本套衣服今日主要活動範圍', '上衣', '外套', '褲子 / 裙子', '為了因應天氣所配戴的配件','你覺得今天的穿著適中嗎？']

# 建立數據處理流水線
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

# 合併數值和類別欄位的處理
preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, numerical_cols),
                  ('cat', categorical_transformer, categorical_cols)])

# 使用流水線對數據進行轉換
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# 處理後的數據
X_transformed = pipeline.fit_transform(df)

# 如果 X_transformed 是稀疏矩陣，轉換為密集數組
X_transformed_dense = X_transformed.toarray()  # 將稀疏矩陣轉為密集矩陣

persist_directory  = './my_database'  # 這將把 .db 檔案儲存在當前目錄下的 my_database 資料夾
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    # 遍歷資料夾內的所有檔案和子資料夾
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # 刪除檔案或子資料夾
        if os.path.isfile(file_path):
            os.remove(file_path)  # 刪除檔案
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # 刪除子資料夾

    print(f"{folder_path} 資料夾的內容已被清空，但資料夾本身保留")
else:
    print(f"{folder_path} 資料夾不存在")
# 初始化 Chroma 資料庫
client = chromadb.PersistentClient(path=persist_directory )
collections = client.list_collections()

collection = client.create_collection("my_collection")


# 假設你有一些元數據，這裡是從你的資料中提取的每一行數據
#test: documents = df['性別'].astype(str).tolist()  # 使用性別作為文件的示例標識
documents = df[['穿著本套衣服時的溫度(°C)', '穿著本套衣服時的體感溫度 (°C)', '穿著本套衣服時的相對濕度(%)', 
                '性別', '本套衣服今日主要活動範圍', '上衣', '外套', '褲子 / 裙子', '為了因應天氣所配戴的配件',
                '你覺得今天的穿著適中嗎？']].astype(str).agg(' '.join, axis=1).tolist()  # 將這些欄位合併為一個字符串
metadatas = [{'index': i} for i in range(len(df))]  # 這裡用 index 作為每條數據的元數據

# 為每條數據生成唯一的 ID（使用 UUID）
ids = [str(uuid.uuid4()) for _ in range(len(df))]

# 插入處理過的數據到 Chroma 向量資料庫
collection.add(
    documents=documents,
    metadatas=metadatas,
    embeddings=X_transformed_dense.tolist(),  # 使用密集格式的數據
    ids=ids  # 加入每條數據的唯一 ID
)


# 查詢資料
query_data = {
    '性別': ['男'],
    '本套衣服今日主要活動範圍': ['室內'],
    '上衣': ['短袖'],
    '外套': ['無'],
    '褲子 / 裙子': ['牛仔褲'],
    '為了因應天氣所配戴的配件': ['無'],
    '你覺得今天的穿著適中嗎？': ['是，穿的剛剛好，不會太熱也不會太冷'],
    '穿著本套衣服時的溫度(°C)': [22],
    '穿著本套衣服時的體感溫度 (°C)': [21],
    '穿著本套衣服時的相對濕度(%)': [60]
}

# 將查詢資料轉換為 DataFrame
query_df = pd.DataFrame(query_data)

# 使用同樣的處理管道來轉換查詢資料
query_transformed = pipeline.transform(query_df)

# 查詢向量的維度應該是和資料庫中一致的
query_embeddings = query_transformed.toarray().flatten()

# 確保查詢向量的維度
print(f"查詢向量的維度: {len(query_embeddings)}")

# 進行查詢
query_results = collection.query(query_embeddings=query_embeddings, n_results=2)
print(query_results)

result_ids = query_results['ids'][0]  # 查詢結果的 ids
result_metadatas = query_results['metadatas'][0]  # 查詢結果的元數據
result_distances = query_results['distances'][0]  # 查詢結果的距離

# 根據結果中的索引來對應到原始資料
for idx, result_id in enumerate(result_ids):
    metadata = result_metadatas[idx]
    distance = result_distances[idx]  # 提取距離
    data_index = metadata['index']  # 這是元數據中的 index，對應原始資料的索引
    document = documents[data_index]  # 這裡的 documents 是你原始的資料列表
    print(f"Result {idx+1}:")
    print(f"ID: {result_id}")
    print(f"Document: {document}")  # 這裡顯示原始資料的內容（例如性別、溫度等）
    print(f"Metadata Index: {data_index}")
    print(f"Distance: {distance}")  # 輸出距離
    print("-----")