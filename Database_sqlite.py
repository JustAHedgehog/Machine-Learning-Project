import sqlite3
class Database():
  def __init__(self, path):
    self.path = path
    self.con = sqlite3.connect(path)

  def query(self, magic_words):
      try:
          result = self.con.execute(magic_words)
          self.con.commit()
          return list(result)
      except sqlite3.Error as e:
          return f"Error: {e}"
      
  def read(self, table_name, condition:list[str,str]=None)->list:
      cursor = f"SELECT * from {table_name}"
      if condition:
          cursor += f" where {condition[0]} = '{condition[1]}'"
      cursor = self.query(cursor)
      for i in range(len(cursor)):cursor[i] = list(cursor[i])
      return cursor

  def read_title(self, table_name)->list:
      cursor = f"PRAGMA table_info ({table_name})"
      cursor = self.query(cursor)
      result = []
      for data in cursor:
         result.append(data[1])
      return result
  
  def update(self, table_name:str, vacancy:str, profession:str):
      return self.query(f"""UPDATE {table_name} SET
      職業 = '{profession}'
      WHERE 職缺名稱 = '{vacancy}';""")
  """
  update("job_104", "機構工程師-擴編", "機械設計工程師") => 將職缺【機構工程師-擴編】補上職業【機械設計工程師】
  """
        

if __name__ == "__main__":
  # read = Database(path="./job.db")
  # print(read.read(table_name='competency'))
  # print(read.table(table_name='competency', condition=['職業別', '電機工程技術員']))

  test = Database(path="./job.db")
  # edit.update_table("job_104", "擴線需求探針卡(Probe Card)測試副工程師工程師 (依學經歷核定職稱)精材科技股份有限公司_9/13(五)實體媒合會現場面談", "醫療器材產業醫療電子器材研發工程師")
  # edit.update("job_104", "擴線需求探針卡(Probe Card)測試副工程師工程師 (依學經歷核定職稱)精材科技股份有限公司_9/13(五)實體媒合會現場面談", "")
  print(test.read_title(table_name="competency"))