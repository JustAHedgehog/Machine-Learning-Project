# 匯入資料庫
import reference.Database_chroma as Database_chroma
import Database_sqlite

# 輸入正確資料庫路徑
sqlite_db = Database_sqlite.Database(path="job.db")
chroma_db = Database_chroma.Database(path="./Database")


# sqlite資料庫使用範例
sqlite_db.read(table_name="job_104", condition=["職業","RF研發工程師"])
'''
read: 讀取sqlite資料庫
    table_name: job_104, competency   table名稱
    condition: ["str1","str2"]        過濾條件，可選
    會執行sql語法:  SELECT * FROM table_name WHERE str1 = 'str2'

output: list[list[str]]

example:
    sqlite_db.read(table_name="competency")
    sqlite_db.read(table_name="job_104", condition=["職業","RF研發工程師"])
    sqlite_db.read(table_name="job_104", condition=["職缺名稱","realme_手機測試工程師"])
'''


sqlite_db.read_title(table_name="competency")
'''
read_title: 讀取table所有column的標題
    table_name: job_104, competency   table名稱

output: list[str]

example:
    sqlite_db.read_title(table_name="job_104")
    sqlite_db.read_title(table_name="competency")
'''

# chroma向量資料庫使用範例
chroma_db.query(
  table_name="job_104",
  key_words=["機械系畢業，專精2D/3D繪圖軟體與建模軟體"],
  n_results=3,
  # where={"職業":{"$in":["測量放樣助理","振動分析工程人員"]}},
  include=["metadatas", "documents","distances"]
)
'''
query: 使用最近鄰搜尋n筆資料
    table_name: job_104, competency                     table名稱
    key_words: ["sentence1","sentence2","sentence3"]    要拿來搜索的文字
    n_results: int                                      回傳n筆資料
    where:
        用來過濾metadatas
        {"職業":"LED光學設計工程師"}                         僅有職業為"LED光學設計工程師"會被放進搜索
        {"職業":{"$in":["測量放樣助理","振動分析工程人員"]}}  僅有職業在list裡面才會被放進搜索
        更多寫法:https://docs.trychroma.com/guides#using-inclusion-operators-(-and-)
    include:
        設定搜索到的資料會包含什麼資訊，預設為["metadatas","documents"]
        ["documents", 文本原始內容(str)
        "embeddings", 文本向量數據(list[float])
        "metadatas",  元數據(dict)
        "distances"]  距離(相似度)

output: dict

example:
    chroma_db.query(
      table_name="job_104",
      key_words=["機械系畢業，專精2D/3D繪圖軟體與建模軟體"],
      n_results=3,
      # where={"職業":{"$in":["測量放樣助理","振動分析工程人員"]}},
      include=["embeddings", "metadatas", "documents","distances"]
    )
'''

chroma_db.query(
  table_name="competency",
  key_words=["職缺名稱:機構工程師。公司產業類別:電腦及其週邊設備製造業。工作內容:(1) 溝通能力良好，喜歡團隊合作。 (2) 需陪同業務拜訪客戶，並在技術上給予支援。  (3) 協助理解客戶產品需求，並即時將相關訊息提供給工廠RD。 (4) 針對新開發案，撰寫並提供客戶正確的產品相關訊息。 (5) 回覆客戶的技術問答，提供現有解決方案。"],
  n_results=3,
  include=["distances"]
)
# 醫療設備工程管理人員
