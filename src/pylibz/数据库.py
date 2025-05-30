

def neo4j():
    return """

    8RtFT16JyyjS1j6naU24GY4VSjmpI9MKpp7TlVNf1a0
    安装数据库
1 安装JDK
2 安装neo4j

# 拉取docker
docker run --publish=7474:7474 --publish=7687:7687 -e 'NEO4J_AUTH=neo4j/12345678' neo4j
"""







```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.tools.brave_search import BraveSearchToolSpec
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.llms import ChatMessage, MessageRole
import json
import os


def get_query_engine_for_fastapi(index,namespace):
    pc = Pinecone(api_key=os.environ.get('pinecone_key'))
    pinecone_index = pc.Index(index)
    # 初始化 PineconeVectorStore
    FastAPI_vector_store = PineconeVectorStore(pinecone_index=pinecone_index,namespace=namespace)
    # 创建 VectorStoreIndex
    fastapi_index = VectorStoreIndex.from_vector_store(FastAPI_vector_store)
    query_engine = fastapi_index.as_query_engine()
    return query_engine

```


```python
    query_engine = get_query_engine_for_fastapi(index = "transaction-index",namespace = "test1")
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="transactions_index",
            description="It stores some transaction contract data on the blockchain, and you can get information about a certain transaction through this tool"
        ))
    agent = ReActAgent.from_tools(
        tools=[query_engine_tool] + tool_spec.to_tool_list(), verbose=True
    )



```


```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.readers.base import BaseReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Settings
from llama_index.core import Document
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import pandas as pd
import os


class TransactionCSVReader(BaseReader):
    def load_data(self, file_path: str):
        margin = pd.read_csv(file_path)
        documents = []
        for i in margin.groupby('hash.1'):
            documents.append(Document(text = str(i[1].iloc[0].to_dict()),metadata={'transaction':i[0]}))
        return documents

class BlockCSVReader(BaseReader):
    def load_data(self, file_path: str):
        margin = pd.read_csv(file_path)
        documents = []
        for i in margin.groupby('hash'):
            documents.append(Document(text = str(i[1].to_dict()),metadata={'block':i[0]}))
        return documents



pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.expand_frame_repr', False)  # 设置不折叠数据
pd.set_option('display.max_colwidth', None)  # 设置列宽为无限制


csv_dir = ''
index = "transaction-index"
namespace = "test1"
temperature = 0.1



text_splitter = SentenceSplitter(chunk_size=8000, chunk_overlap=50)

api_key = os.environ.get('OPENAI_API_KEY')
llm = OpenAI(
    model="gpt-4o",
    api_key=api_key,
    temperature=temperature,
    # max_tokens=None,
)
embed_model = OpenAIEmbedding(api_key=api_key)
Settings.embed_model = embed_model
Settings.llm = llm

pipeline = IngestionPipeline(
    transformations=[text_splitter]
)
# 自定义reader

pc = Pinecone(api_key=os.environ.get('pinecone_key'))
pinecone_index = pc.Index(index)
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, namespace=namespace
)
# 创建StorageContext来管理 vector_store
storage_context = StorageContext.from_defaults(vector_store=vector_store)

reader = TransactionCSVReader()
documents = reader.load_data(csv_dir)
nodes = pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True,
)

index = VectorStoreIndex(nodes, storage_context=storage_context)



```





方便使用的向量数据库


安装
```
!pip install pinecone-client
!pip install pinecone
```



```python
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
```

## 创建Index
```python
pc.create_index(
    "api-documents-index",
    dimension=1536, # 维度1536
    metric="euclidean",#欧式空间 "cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
```

## 获取Index

```python
pinecone_index = pc.Index("your_index_name")
```

## 删除Index
```python
pc.delete_index("quickstart-index")
```





# 安装


```bash
brew install postgresql
```

# 创建数据库

```
CREATE DATABASE vector_db  创造数据库
```



如何连接到数据库


安装
```
!pip install pandas psycopg2-binary
```

psycopg2 连接数据库
```python
PGInfo = MatchNewPG
import psycopg2
try:
    connection = psycopg2.connect(
        host=PGInfo.host,
        database=PGInfo.dbname,
        user=PGInfo.user,
        password=PGInfo.password,
        port=PGInfo.port,
    )
    print("Connection successful")
except psycopg2.OperationalError as e:
    print(f"Connection failed: {e}")

```


```python
cursor = connection.cursor() # 获取游标
```

关闭连接
```python

cursor.close() #关闭游标
connection.close() # 关闭连接
```



```python
from sqlalchemy import create_engine

def df_from_sql(uri,dbname,sql_query = None):
    uri_new = f"{uri}/{dbname}"
    engine = create_engine(uri_new)
    sql_query = sql_query or f"""
    SELECT * FROM public.blocks WHERE block_number = {block_number} LIMIT 10000;
    """
    df_ = pd.read_sql(sql_query, engine)
    return df_
    
```





数据库-SQL查询
```
SELECT t.*, CTID  
FROM public.transactions t  
LIMIT 500001
```

```
SELECT * FROM public.blocks t1 JOIN public.transactions t2 ON t1.hash = t2.block_hash LIMIT 1000;
```

```
SELECT * FROM public.blocks t1 JOIN public.transactions t2 ON t1.hash = t2.block_hash
LIMIT 1000 OFFSET 0; -- 第一批数据

SELECT * FROM public.blocks t1 JOIN public.transactions t2 ON t1.hash = t2.block_hash
LIMIT 1000 OFFSET 1000; -- 第二批数据

SELECT * FROM public.blocks t1 JOIN public.transactions t2 ON t1.hash = t2.block_hash
LIMIT 1000 OFFSET 2000; -- 第三批数据
```



```python
SELECT *
FROM public.token_trades
ORDER BY trade_time DESC
LIMIT 1000;
```


SELECT
    block_number,
    hash,
    from_address,
    to_address,
    gas_price,
    gas_used
FROM
    public.transactions
WHERE
    from_address = '0x5EB35DADF754F8EC9FFC29F4DB149EAAA9EC50E2';




```python

import redis

r = redis.Redis(host='localhost', port=6379, db=0)

# 设置缓存
r.set('foo', 'bar')

# 获取缓存
value = r.get('foo')

```


