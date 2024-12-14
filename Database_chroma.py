import chromadb
from FlagEmbedding import BGEM3FlagModel
class Database():
  def __init__(self, path):
    self.client = chromadb.PersistentClient(path=path)
    self.job_104 = self.client.get_or_create_collection("job_104")
    self.competency = self.client.get_or_create_collection("competency")
    self.model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
  
  def _get_collection(self, table_name):
    if table_name == "job_104":
      return self.job_104
    elif table_name == "competency":
      return self.competency
    else:
      raise ValueError(f"Invalid table_name: {table_name}")
    
  def _embedding(self, data):
    data_after_embedding = self.model.encode(
      data,
      return_dense=True,
      return_sparse=False,
      return_colbert_vecs=False
    )
    return data_after_embedding["dense_vecs"]

  def add(self, table_name, ids:list, metadatas:list[dict], documents:list[str])->None:
    table = self._get_collection(table_name)
    table.add(
      ids=ids,
      embeddings=self._embedding(documents),
      metadatas=metadatas,
      documents=documents,
    )
  
  def query(self, table_name:str, key_words:list[str], n_results:int=5, where:dict=None, include:list=["metadatas","documents"])->dict:
    table = self._get_collection(table_name)
    encode = self._embedding(key_words)
    return table.query(
      query_embeddings=encode,  # 查詢語句之向量
      n_results=n_results,      # 返回n項結果
      where=where,              # 篩選metadata，假設where={"職缺名稱":{"$in":["職缺1","職缺2"]}}
      include=include           # 返回之dict包含的資訊include=["embeddings", "metadatas", "documents"]，預設為["metadatas","documents"]
      )
# where寫法可參見https://docs.trychroma.com/guides#using-inclusion-operators-(-and-)


if __name__ == "__main__":
  test = Database("./Database")
  test.query("competency", ["軟體工程師","機構工程師"])

  test = chromadb.PersistentClient("./Database")
  test.delete_collection("competency")
  test.delete_collection("job_104")