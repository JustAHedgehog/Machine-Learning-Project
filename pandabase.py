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

# 定義輸入特徵
input_cols = ['現在穿著本套衣服時的溫度(°C)', '現在穿著本套衣服時的體感溫度 (°C)', 
              '穿著本套衣服時的相對濕度(%)', '性別', '本套衣服今日主要活動範圍']
# 定義輸出 (穿著和配件)
output_cols = ['上衣', '外套', '褲子 / 裙子', '為了因應天氣所配戴的配件']

# 適中性轉換表
rating_mapping = {'是，穿的剛剛好，不會太熱也不會太冷': 1, '否，我覺得我穿太少了，應該要多加幾件': -1, '否，穿太厚了，在室內沒開冷氣很熱': -1}
df['適中性'] = df['你覺得今天的穿著適中嗎？'].map(rating_mapping)

# 數據預處理流水線
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[('num', numerical_transformer, ['穿著本套衣服時的溫度(°C)', 
                                                  '穿著本套衣服時的體感溫度 (°C)', 
                                                  '穿著本套衣服時的相對濕度(%)']),
                  ('cat', categorical_transformer, ['性別', '本套衣服今日主要活動範圍'])]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_inputs = pipeline.fit_transform(df[input_cols])

# 如果資料夾存在，清空
folder_path = "my_database"
if os.path.exists(folder_path) and os.path.isdir(folder_path):
    shutil.rmtree(folder_path)

# 初始化 Chroma 資料庫
persist_directory = './my_database'
client = chromadb.PersistentClient(path=persist_directory)
collection = client.create_collection("outfit_recommendations")

# 輸出文檔和元數據
documents = df[output_cols].astype(str).agg(' '.join, axis=1).tolist()
metadatas = df[input_cols + ['適中性']].to_dict(orient='records')
ids = [str(uuid.uuid4()) for _ in range(len(df))]

# 插入到 Chroma 資料庫
collection.add(
    documents=documents,
    metadatas=metadatas,
    embeddings=X_inputs.toarray().tolist(),
    ids=ids
)

print(f"插入了 {len(df)} 條資料進入 Chroma 資料庫")

# 查詢：條件 (輸入溫度、性別等)
query_data = {
    '穿著本套衣服時的溫度(°C)': [22],
    '穿著本套衣服時的體感溫度 (°C)': [21],
    '穿著本套衣服時的相對濕度(%)': [60],
    '性別': ['男'],
    '本套衣服今日主要活動範圍': ['室內']
}
query_df = pd.DataFrame(query_data)
query_inputs = pipeline.transform(query_df)
query_embeddings = query_inputs.toarray().flatten()

# 查詢 Chroma 資料庫
query_results = collection.query(query_embeddings=query_embeddings, n_results=5)

# 過濾結果 (移除「穿太多」或「穿太少」的結果)
filtered_results = [
    (doc, meta, dist) for doc, meta, dist in 
    zip(query_results['documents'], query_results['metadatas'], query_results['distances'])
    if meta['適中性'] == 1  # 只保留「適中」的結果
]

# 顯示結果
for idx, (doc, meta, dist) in enumerate(filtered_results):
    print(f"Result {idx + 1}:")
    print(f"Recommended Outfit: {doc}")
    print(f"Query Metadata: {meta}")
    print(f"Similarity Distance: {dist}")
    print("-----")

if not filtered_results:
    print("無適中推薦結果，請重新調整輸入條件。")
