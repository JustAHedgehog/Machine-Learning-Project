import streamlit as st
import google.generativeai as genai
import chromadb
import database_chroma as chroma


# 配置 Google Gemini API
api_key = ''
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# 加載 Chroma 資料庫
persist_directory = './my_database'  # 資料庫目錄
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("outfit_recommendations")

# Streamlit 界面設計
st.title("AI 穿著建議系統")
st.subheader("請提供當前條件以獲取最佳穿著建議：")

# 用戶查詢條件
with st.form("user_query_form"):
    temperature = st.number_input('現在的溫度 (°C)', min_value=-50, max_value=50, step=1)
    body_temp = st.number_input('現在的體感溫度 (°C)', min_value=-50, max_value=50, step=1)
    humidity = st.number_input('現在的相對濕度 (%)', min_value=0, max_value=100, step=1)
    gender = st.radio("性別", ["男", "女"])
    zone = st.selectbox("今日主要活動範圍", ["室外", "室內"])
    submitted = st.form_submit_button("生成穿著建議")

# 當用戶提交表單時進行處理
if submitted:
    user_query = {
        "temperature": int(temperature),
        "body_temp": int(body_temp),
        "humidity": int(humidity),
        "gender": gender,
        "zone": zone
    }

    # 調用 Chroma 檢索結果
    retrieval_results = chroma.get_query(
        user_query["temperature"],
        user_query["body_temp"],
        user_query["humidity"],
        user_query["gender"],
        user_query["zone"],
        chroma.pipeline,
        collection
    )

    # 構建上下文
    retrieved_context = \
        f"資料:\n性別 {user_query['gender']}，溫度 {user_query['temperature']}°C，體感 {user_query['body_temp']}°C，相對濕度 {user_query['humidity']}%，" + \
        f"活動範圍 {user_query['zone']}\n"
    for i, docs in enumerate(retrieval_results['documents'][0]):
        doc = docs.split()
        for content in doc:
            if "," in content:
                index = doc.index(content)
                doc[index] += doc[index + 1]
                doc.pop(index + 1)
        retrieved_context += f"建議穿著 {i+1}：{doc}\n"

    # 發送至 Google Gemini
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
    response = model.generate_content(prompt)

    # 顯示結果
    st.subheader("AI 穿著建議：")
    st.write(response.result)
