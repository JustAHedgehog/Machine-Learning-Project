from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import chromadb
import uuid
import os
import shutil


def dataPreprocess():
    # 數據預處理流水線
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[('num', numerical_transformer, ['現在穿著本套衣服時的溫度(°C)',
                                                      '現在穿著本套衣服時的體感溫度 (°C)',
                                                      '穿著本套衣服時的相對濕度(%)']),
                      ('cat', categorical_transformer, ['性別', '本套衣服今日主要活動範圍'])]
    )
    return preprocessor

def query(temp:int, body_temp:int, humidity:int, gender:str, zone:str):
    query_data = {
        '現在穿著本套衣服時的溫度(°C)': [temp],
        '現在穿著本套衣服時的體感溫度 (°C)': [body_temp],
        '穿著本套衣服時的相對濕度(%)': [humidity],
        '性別': [gender],
        '本套衣服今日主要活動範圍': [zone]
    }
    return query_data

if __name__ == "__main__":
    # 讀取CSV數據
    df = pd.read_csv('data.csv')
    df = df.drop(columns=['時間戳記', '電子郵件'])

    # 定義輸入輸出特徵
    input_cols = ['現在穿著本套衣服時的溫度(°C)', '現在穿著本套衣服時的體感溫度 (°C)',
                '穿著本套衣服時的相對濕度(%)', '性別', '本套衣服今日主要活動範圍']
    output_cols = ['上衣', '外套', '褲子 / 裙子', '為了因應天氣所配戴的配件']

    # 適中性轉換表
    rating_mapping = {'是，穿的剛剛好，不會太熱也不會太冷': 1,
                    '否，我覺得我穿太少了，應該要多加幾件': -1, '否，穿太厚了，在室內沒開冷氣很熱': -1}
    df['適中性'] = df['你覺得今天的穿著適中嗎？'].map(rating_mapping)

    pipeline = Pipeline(steps=[('preprocessor', dataPreprocess())])
    X_inputs = pipeline.fit_transform(df[input_cols])

    # ============
    # 初始化 Chroma 資料庫
    folder_path = "my_database"
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

    persist_directory = './my_database'
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.create_collection("outfit_recommendations")

    # 輸出文檔和元數據，正常數量 = 175
    documents = df[output_cols].astype(str).agg(' '.join, axis=1).tolist()
    metadatas = df[input_cols + ['適中性']].to_dict(orient='records')
    ids = [str(uuid.uuid4()) for _ in range(len(df))]
    embeddings = X_inputs.tolist()

    # 插入到 Chroma 資料庫
    collection.add(
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
        ids=ids
    )

    # 測試功能
    query_data = query(22,21,60,'男','室內')
    query_df = pd.DataFrame(query_data)
    query_inputs = pipeline.transform(query_df)

    query_embeddings = query_inputs.flatten()  # 將向量拉直
    query_results = collection.query(
        query_embeddings=query_embeddings, n_results=3)

    # 確保結果處理與篩選
    filtered_results = [
        (doc, meta, dist) for doc, meta, dist in
        zip(query_results['documents'],
            query_results['metadatas'], query_results['distances'])
        if meta[0]['適中性'] >= 0  # 過濾適中性結果
    ]

    # 顯示結果
    if filtered_results:
        for idx, (doc, meta, dist) in enumerate(filtered_results):
            print(f"Result {idx + 1}:")
            print(f"Recommended Outfit: {doc}")
            print(f"Query Metadata: {meta}")
            print(f"Similarity Distance: {dist}")
            print("-----")
    else:
        print("無適中推薦結果，請重新調整輸入條件。")
