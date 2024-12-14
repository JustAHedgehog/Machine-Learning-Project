from openai import OpenAI
from typing import Dict, Generator
import ollama
import streamlit as st
import Database_sqlite
import Database_chroma
import re

def ollama_generator(messages:Dict) -> Generator:
    stream = ollama.chat(
        model='llama3.1',
        messages=messages,
        stream=True
    )
    for chunk in stream:
        yield chunk['message']['content']

def openai_generate(messages:Dict):
    stream = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

@st.cache_resource
def get_database_sqlite(path):
    return Database_sqlite.Database(path=path)
@st.cache_resource
def get_database_chroma(path):
    return Database_chroma.Database(path=path)

st.set_page_config(page_title="大學生就業指南")
st.title("大學生就業指南")
st.subheader("通過對話來發現你的職業內涵")

db_sqlite = get_database_sqlite(path="./job.db")
db_chroma = get_database_chroma(path="./Database")
client = OpenAI(api_key=' ')


###################################################################################
def initial():
  k = []
  s = []

  datas = db_sqlite.read(
    table_name="competency",
  )
  for i in range(len(datas)):
    k += (datas[i][5].split(", "))
    s += (datas[1][6].split(", "))

  k = list(dict.fromkeys(k))
  s = list(dict.fromkeys(s))


  k_prompts = ""
  s_prompts = ""
  for k_prompt in k:
    k_prompts += f"- {k_prompt}\n"
  for s_prompt in k:
    s_prompts += f"- {s_prompt}\n"

  conversation_history = []

  prompt = f'''
  你是一位職業顧問，專注於幫助使用者發現他們的職業內涵，包括職業內涵k(知識)和職業內涵s(技能)。請根據使用者的回答來判斷他們擁有的職能。以下是職能的敘述與列表：

  職能內涵k(知識)：指執行某項任務所需瞭解可應用於該領域的原則與事實。
  職能內涵s(技能)：指執行某項任務所需具備可幫助任務進行的認知層面能力或技術性操作層面的能力(通稱hard skills)，以及跟個人有關之社交、溝通、自我管理行為等能力(通稱soft skills)。
  '''
  conversation_history.append({"role":"system","content":prompt})

  prompt = f'''
  **職業內涵k(知識)**：
  {k_prompts}

  **職業內涵s(技能)**：
  {s_prompts}
  '''
  conversation_history.append({"role":"system","content":prompt})

  prompt = f'''
  請開始與使用者進行對話，通過提出一系列開放式問題來了解他們的背景、興趣和技能。請注意，每次只提出一個問題，並在等待使用者回答後再提出下一個問題。請確保在提出3個問題之後，根據使用者的回答，自行判斷出他們可能擁有的職業內涵k和s，並確保這些職能僅限於上述列表中。

  在你確定了使用者的職能後，請按照以下格式輸出結果，且確保每個類別最多不超過10個項目：

  ### 職業內涵k(知識):
  - 知識1
  - 知識2
  - ...

  ### 職業內涵s(技能):
  - 技能1
  - 技能2
  - ...

  並在最後附上「判斷已完成，對話結束。」以表示對話的結束。

  在開始對話時，可以用友好且引導性的語言，讓使用者感到舒適並鼓勵他們分享更多。請注意全程使用中文進行應答。請開始對話：
  '''
  conversation_history.append({"role":"system","content":prompt})
  return conversation_history
###################################################################################


###################################################################################
def report_data(ai_response):
  # 使用正則表達式提取 k 和 s
  k_match = re.search(r'### 職業內涵k\(知識\):\n((?:- .*\n?)+)', ai_response)
  s_match = re.search(r'### 職業內涵s\(技能\):\n((?:- .*\n?)+)', ai_response)

  # 提取並格式化
  k_values = [item.strip('- ').strip() for item in k_match.group(1).strip().split("\n")] if k_match else []
  s_values = [item.strip('- ').strip() for item in s_match.group(1).strip().split("\n")] if s_match else []

  # 打印結果
  print("職業內涵k(知識)列表:", k_values)
  print("職業內涵s(技能)列表:", s_values)

  query = "職業內涵k: "
  for i in range(len(k_values)):
    query += f"{k_values[i]}"
    if k_values[i] != k_values[-1]:
      query += ", "
    else:
      query += "。"
  query += "職業內涵s: "
  for i in range(len(s_values)):
    query += f"{s_values[i]}"
    if s_values[i] != s_values[-1]:
      query += ", "
    else:
      query += "。"

  # db_chroma = Database_chroma.Database(path="./Database")

  matched_jobs = db_chroma.query(
    table_name="competency",
    key_words=[query],
    n_results=3
    )["ids"][0]

  job_summary = db_chroma.query(
    table_name="job_104",
    key_words=[query],
    n_results=3,
    where={"職業":{"$in":matched_jobs}}
    )["ids"][0]

  jobs = db_sqlite.query(
    magic_words=f"select * from job_104 where 職缺名稱 in {tuple(job_summary)}"
  )
  title = db_sqlite.read_title(table_name="job_104")
# '''
# 0   職缺名稱
# 1   工作地點
# 2   學歷要求
# 3   工作經歷
# 4   科系要求
# 5   公司名稱
# 6   公司產業類別
# 7   工作待遇
# 8   更新日期
# 9   工作內容
# 10  應徵分析
# 11  職缺連結
# 12  公司連結
# 13  職業
# '''
  vacancies = []
  for job in jobs:
    vacancy = ""
    for i in range(len(title)):
      if i in [0,1,2,3,4,5,6,7,9,13]:
        vacancy += f"{title[i]}: {job[i]}。"
    vacancies.append(vacancy)

  conversation_history = [st.session_state.messages[0]]+st.session_state.messages[3:]

  prompt = f'''
你是一位職業分析師，根據使用者的回答、職能分析結果以及從向量資料庫中查找到的職業和職缺資料，請生成一份職能分析報告。報告應該包括以下內容：
 **使用者擁有的職能**：
   - 使用者擁有的知識 (k)
   - 使用者擁有的技能 (s)
'''
  conversation_history.append({"role":"system","content":prompt})

  prompt = f'''
 **對應的職業**：
   根據使用者的職能分析，以下是最相關的職業：
   - 職業1：{matched_jobs[0]}
   - 職業2：{matched_jobs[1]}
   - 職業3：{matched_jobs[2]}
'''
  conversation_history.append({"role":"system","content":prompt})

  prompt = f'''
 **推薦的職缺概述**：
   最後，這裡是針對上述職業的推薦職缺的簡短概述。這些職缺的詳細資料如公司名稱、上班地點、職缺連結等，將在使用者進一步詢問時提供：
   - 職缺1：{vacancies[0]}
   - 職缺2：{vacancies[1]}
   - 職缺3：{vacancies[2]}
請根據以上信息，生成一份詳細的職缺及職能分析報告，請注意包含推薦的職缺概述，並且明確告知使用者可以要求更多詳細資料。詳細資料包括但不限於公司名稱、上班地點、職缺連結等。
'''
  conversation_history.append({"role":"system","content":prompt})

  prompt = f'''
請根據以上信息，生成一份詳細的職缺及職能分析報告，*請使用此模板來生成報告*
【分析報告】

### 1. 你所擁有的職能：
- **職業內涵(知識k)：** 
  - 內涵1
  - 內涵2
  ...

- **職業內涵(技能s)：**
  - 內涵1
  - 內涵2
  ...

### 2. 對應的職業：
根據使用者的職能分析，以下是最相關的職業：
  - **職業1：{matched_jobs[0]}** 
  - **職業2：{matched_jobs[1]}** 
  - **職業3：{matched_jobs[2]}** 

### 3. 推薦的職缺概述：
以下是針對上述職業的推薦職缺的概述。注意：這裡只提供簡要概述，但職缺名稱需與匹配之職缺相同，如需詳細信息（包括公司名稱、上班地點、工作待遇、應徵分析、職缺連結、公司連結等），請使用者進一步詢問。注意:**應徵分析**、
**職缺連結**、**公司連結**這三項必須等使用者提問才需要特別告知使用者。
- **職缺1：** 
  - **職缺名稱**: {jobs[0][0]}
  - **學歷要求**: {jobs[0][2]}
  - **工作經歷**: {jobs[0][3]}
  - **科系要求**: {jobs[0][4]}
  - **工作內容**: {jobs[0][9]}
  - **應徵分析**: {jobs[0][10]}
  - **職缺連結**: {jobs[0][11]}
  - **公司連結**: {jobs[0][12]}
- **職缺2：**
  - **職缺名稱**: {jobs[1][0]}
  - **學歷要求**: {jobs[1][2]}
  - **工作經歷**: {jobs[1][3]}
  - **科系要求**: {jobs[1][4]}
  - **工作內容**: {jobs[1][9]}
  - **應徵分析**: {jobs[1][10]}
  - **職缺連結**: {jobs[1][11]}
  - **公司連結**: {jobs[1][12]}
- **職缺3：**
  - **職缺名稱**: {jobs[2][0]}
  - **學歷要求**: {jobs[2][2]}
  - **工作經歷**: {jobs[2][3]}
  - **科系要求**: {jobs[2][4]}
  - **工作內容**: {jobs[2][9]}
  - **應徵分析**: {jobs[2][10]}
  - **職缺連結**: {jobs[2][11]}
  - **公司連結**: {jobs[2][12]}
承接以上，如使用者詢問連結請嚴格依照以下格式並以markdown語法輸出:
 - **應徵分析**:
 - **職缺連結**:
 - **公司連結**:

### 4. 結論與建議：
請根據以上分析提供一個總結，並給出職業發展的建議。提醒使用者可以進一步詢問以獲得更詳細的職缺資料。
'''
  conversation_history.append({"role":"system","content":prompt})

  conversation_history.append({"role":"user","content":"請產出分析報告"})

  return conversation_history
###################################################################################


if "messages" not in st.session_state:
  st.session_state.messages = initial()
  st.session_state.report = ""
  st.session_state.selected_model = "gpt-4o-mini"
  st.session_state.start_report = False
  with st.chat_message("ai"):
    ai_response = st.write_stream(openai_generate(st.session_state.messages))
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
else:
  for message in st.session_state.messages:
      if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if user_input := st.chat_input("Say something..."):

    if st.session_state.selected_model == "gpt-4o-mini":
      st.session_state.messages.append({"role": "user", "content": user_input})
      with st.chat_message("user"):
          st.markdown(user_input)
      ai_response=""    
      with st.chat_message("assistant"):
          ai_response = st.write_stream(openai_generate(st.session_state.messages))
      st.session_state.messages.append({"role": "assistant", "content": ai_response})
      # 切換到報告分析
      if "判斷已完成，對話結束。" in ai_response:
         st.session_state.selected_model = "llama3.1"
         st.session_state.messages = report_data(ai_response)
         if st.session_state.start_report == False:
            with st.chat_message("assistant"):
                ai_response = st.write_stream(ollama_generator(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.session_state.report = ai_response
            with st.sidebar:
                st.markdown("## 您的職能分析報告")
                st.markdown(st.session_state.report)
            st.session_state.start_report = True

    # 報告分析環節
    elif st.session_state.selected_model == "llama3.1":
        with st.sidebar:
            st.markdown("## 您的職能分析報告")
            st.markdown(st.session_state.report)
        if st.session_state.start_report:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            ai_response=""    
            with st.chat_message("assistant"):
                ai_response = st.write_stream(ollama_generator(st.session_state.messages))
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
           
