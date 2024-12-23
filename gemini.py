import google.generativeai as genai
import chromadb
import pandas as pd
import database_chroma


# 加載已建立的 Chroma 資料庫
persist_directory = './my_database'  # 存放資料庫的目錄
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("outfit_recommendations")

# Step 2: 用戶查詢條件
user_query = database_chroma.query(28,30,70,"女","室外")

# Step 3: 資料處理，確保一致性
numerical_cols = ['現在穿著本套衣服時的溫度(°C)', '現在穿著本套衣服時的體感溫度 (°C)', '穿著本套衣服時的相對濕度(%)']
categorical_cols = ['性別', '本套衣服今日主要活動範圍']

# 初始化數值與類別數據轉換器
preprocessor = database_chroma.dataPreprocess()

# 將查詢條件轉換為向量
input_data = pd.DataFrame([{
    '現在穿著本套衣服時的溫度(°C)': user_query['現在穿著本套衣服時的溫度(°C)'],
    '現在穿著本套衣服時的體感溫度 (°C)': user_query['現在穿著本套衣服時的體感溫度 (°C)'],
    '穿著本套衣服時的相對濕度(%)': user_query['穿著本套衣服時的相對濕度(%)'],
    '性別': user_query['性別'],
    '本套衣服今日主要活動範圍': user_query['本套衣服今日主要活動範圍']
}])
input_vector = preprocessor.transform(input_data)

# Step 4: 從 Chroma 資料庫檢索相似紀錄
retrieval_results = collection.query(
    query_embeddings=input_vector[0],
    n_results=3  # 最相關的 3 筆資料
)

# Step 5: 構建上下文
retrieved_context = "\n".join([
    f"資料 {i+1}: 性別 {doc['性別']}，溫度 {doc['現在穿著本套衣服時的溫度(°C)']}°C，體感 {doc['現在穿著本套衣服時的體感溫度 (°C)']}°C，"
    f"活動範圍 {doc['本套衣服今日主要活動範圍']}，建議穿著：{doc['推薦穿著']}"
    for i, doc in enumerate(retrieval_results['documents'])
])

# Step 6: 發送到 Gemini 並生成結果
prompt = f"""
你是一位專業的穿著建議專家。根據以下條件，請提供用戶的穿著建議：
- 性別：{user_query['性別']}
- 現在的溫度：{user_query['現在穿著本套衣服時的溫度(°C)']}
- 體感溫度：{user_query['現在穿著本套衣服時的體感溫度 (°C)']}
- 相對濕度：{user_query['穿著本套衣服時的相對濕度(%)']}
- 活動範圍：{user_query['本套衣服今日主要活動範圍']}

以下是類似條件下的歷史紀錄：
{retrieved_context}

請根據以上資訊提供用戶的最佳穿著建議。
"""
api_key = ''
genai.configure(api_key = api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(prompt)

# Step 7: 輸出結果
print("穿著建議：")
print(response.text)