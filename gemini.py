import google.generativeai as genai
import chromadb
import database_chroma as chroma

# 加載已建立的 Chroma 資料庫
persist_directory = './my_database'  # 存放資料庫的目錄
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("outfit_recommendations")

# Step 2: 用戶查詢條件
user_query = dict()
user_query["temperature"] = int(input('現在溫度(°C)：'))
user_query['body_temp'] = int(input('現在的體感溫度 (°C)：'))
user_query['humidity'] = int(input('現在的相對濕度(%)：'))
user_query['gender'] = input('性別：')
user_query['zone'] = input('今日主要活動範圍（室外/室內）：')

input_cols = ['現在穿著本套衣服時的溫度(°C)', '現在穿著本套衣服時的體感溫度 (°C)',
                '穿著本套衣服時的相對濕度(%)', '性別', '本套衣服今日主要活動範圍']

retrieval_results = chroma.get_query(user_query["temperature"],user_query["body_temp"],user_query['humidity'],user_query['gender'],user_query['zone'],chroma.pipeline,collection)

# Step 3: 構建上下文
retrieved_context = \
    f"資料:\n性別 {user_query['gender']}，溫度 {user_query['temperature']}°C，體感 {user_query['body_temp']}°C，相對溼度 {user_query["humidity"]}，" + \
    f"活動範圍 {user_query['zone']}\n"
for i, docs in enumerate(retrieval_results['documents'][0]):
    doc = docs.split()
    for content in doc:
        if "," in content:
            index = doc.index(content)
            doc[index] += doc[index+1]
            doc.pop(index+1)
    retrieved_context += f"建議穿著{i+1}：{doc}\n" 

# Step 4: 發送到 Gemini 並生成結果
prompt = f"""
你是一位專業的穿著建議專家。根據以下條件，請提供用戶的穿著建議：
- 性別：{user_query['gender']}
- 現在的溫度：{user_query['temperature']}
- 體感溫度：{user_query['body_temp']}
- 相對濕度：{user_query['humidity']}
- 活動範圍：{user_query['zone']}

以下是類似條件下的歷史紀錄：
{retrieved_context}

請根據以上資訊提供用戶的最佳穿著建議。
"""
api_key = ''
genai.configure(api_key = api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content(prompt)

# Step 7: 輸出結果
print("="*9)
print("AI穿著建議：")
print(response.text)