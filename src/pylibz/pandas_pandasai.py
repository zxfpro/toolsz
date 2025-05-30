



# pd.merge(blocks, trans, on='blocks_hash', how='inner') 合并两个df



#i : 我希望看看df 中hash列去重后还剩多少?

#Obsidian : 要查看 DataFrame 中 `hash` 列去重后的唯一值数量，可以使用 Pandas 的 `nunique` 方法。假设你想查看 `trans` DataFrame 中 `blocks_hash` 列去重后的数量，以下是示例代码：

```python
import pandas as pd

# trans DataFrame 示例
trans = pd.DataFrame({
    'blocks_hash': ['hash1', 'hash1', 'hash2', 'hash3', 'hash3'],
    'trans_data': ['trans1', 'trans2', 'trans3', 'trans4', 'trans5']
})

# 计算 blocks_hash 列去重后的唯一值数量
unique_hash_count = trans['blocks_hash'].nunique()

# 打印唯一值数量
print(f"去重后的唯一 hash 数量: {unique_hash_count}")
```

在这个示例中，`nunique()` 方法用于计算 `blocks_hash` 列中唯一值的数量，并将结果存储在 `unique_hash_count` 变量中。然后，我们打印出去重后的唯一 `hash` 数量。success

[[如何连接到数据库]]




进行列方向的合并

```python
pd.concat([df,df2],axis=0)

```

进行行方向的合并

```python
pd.concat([df,df2],axis=1)
```


对应合并
```python
  
air_quality = pd.merge(air_quality, air_quality_parameters,

how='left', left_on='parameter', right_on='id')
```

```python
# 创建两个DataFrame
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# 内连接
merged_inner = pd.merge(df1, df2, on='key', how='inner')
print(merged_inner)  

# 外连接
merged_outer = pd.merge(df1, df2, on='key', how='outer')
print(merged_outer)

# 左连接
merged_left = pd.merge(df1, df2, on='key', how='left')
print(merged_left)

# 右连接
merged_right = pd.merge(df1, df2, on='key', how='right')
print(merged_right)

```

join

```python
# 创建两个DataFrame

df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]}).set_index('key')

df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]}).set_index('key')

  

# 使用join进行连接

joined = df1.join(df2, how='inner')

print(joined)
```




读取CSV

```python
titanic = pd.read_csv("data/titanic.csv")
air_quality = pd.read_csv("data/air_quality_no2.csv", index_col=0, parse_dates=True)
```


读取Excel
```python
titanic = pd.read_excel("titanic.xlsx", sheet_name="passengers")
```
```python
users = pd.read_table('data/users.dat',header=None,names=
['UserID','Gender','Age','Occupation','Zip-code'], sep='::',engine= 'python')

```



保存为Excel
```python
titanic.to_excel("titanic.xlsx", sheet_name="passengers", index=False)
```




一般不建议用rename的函数映射, 可读性不高
```python
# rename做点对点映射
air_quality_renamed = air_quality.rename(
    columns={
        "station_antwerp": "BETR801",
        "station_paris": "FR04014",
        "station_london": "London Westminster",
    }
)

# 使用rename做函数映射
air_quality_renamed = air_quality_renamed.rename(columns=str.lower)

```




```python
对Series的条件判断

ages.isin([23,3])
ages>35
(ages == 22) | (ages == 3)
ages.notna()
ages.str.contains('o')

```



```python
air_quality["datetime"] = pd.to_datetime(air_quality["datetime"])

```

```python
pd.read_csv("../data/air_quality_no2_long.csv", parse_dates=["datetime"])
```


https://pandas.pydata.org/docs/getting_started/intro_tutorials/09_timeseries.html#



```python
dates = pd.date_range("20130101", periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
```


本质上df是一个字典, 所以符合字典的逻辑
```python
air_quality["london_mg_per_cubic"] = air_quality["station_london"] * 1.882

air_quality["ratio_paris_antwerp"] = (
    air_quality["station_paris"] / air_quality["station_antwerp"]
)
```

air_quality["station_paris"] 就是series 而series是列向量, 可以进行常规的列向量运算



如何进行列向量维度的选择
```python
# df[ 被选择的列名列表 ]
df[['Age','Sex']]
```


如何进行行维度的选择
```python
# df[ 是一个对Series的条件判断 ]
df[df['Age']>35]
```

[[对Series的条件判断怎么做]]


创建表（如果尚未创建）
```python

cursor = connection.cursor()

# 创建表（如果尚未创建）
create_table_query = """
CREATE TABLE IF NOT EXISTS employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    department VARCHAR(100)
)
"""
cursor.execute(create_table_query)
connection.commit()
```

使用 Pandas 创建数据
```python

# 使用 Pandas 创建数据
data = {
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "department": ["HR", "IT", "Finance"]
}
df = pd.DataFrame(data)

insert_query = "INSERT INTO employees (name, age, department) VALUES (%s, %s, %s)"
for _, row in df.iterrows():
    cursor.execute(insert_query, tuple(row))


connection.commit()
```

```python

# 查询数据并验证
cursor.execute("SELECT * FROM employees")
rows = cursor.fetchall()
for row in rows:
    print(row)

```

1 使用descirbe
2 使用agg自定义统计类目
```python
df.agg(
	{
	"Age": ["min", "max", "median", "skew"],
	"Fare": ["min", "max", "median", "mean"],
	}
)
```

使用groupby 来进行分组统计

```python
# 分组统计 split-apply-combine模式
df[["Sex", "Age"]].groupby("Sex").mean()
titanic.groupby(["Sex", "Pclass"])["Fare"].mean()

```



```python

#@title 1. Keep this tab alive to prevent Colab from disconnecting you { display-mode: "form" }
%%html

<audio src="https://oobabooga.github.io/silence.m4a" controls> 用来防止发生断裂


#@title String fields
#@markdown Text
value = 'value'  # @param {type:"string"}
text = value

#@markdown Dropdown
value = '1st option'  # @param ["1st option", "2nd option", "3rd option"]
dropdown = value

#@markdown Text and dropdown
value = 'value'  # @param ["1st option", "2nd option", "3rd option"] {allow-input: true}
text_and_dropdown = value

print(text)
print(dropdown)
print(text_and_dropdown)

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

df = pd.DataFrame(
    {
        "Name": [
            "Braund, Mr. Owen Harris",
            "Allen, Mr. William Henry",
            "Bonnell, Miss. Elizabeth",
        ],
        "Age": [22, 35, 58],
        "Sex": ["male", "male", "female"],
    }
)

df

ages = df.Age

df.index, ages.index

df.to_numpy()

df.T

ages.sort_index(axis=0,ascending=True)

ages.sort_values()

# getitem

df[0:3]

df["20130102":"20130104"]





timestamp 时间戳

ts = pd.Series(np.random.randn(1000), index=pd.date_range("1/1/2000", periods=1000))


ts.cumsum()

# df 做定制化的统计 这个可能要开辟代码区

df.agg(
    {
        "Age": ["min", "max", "median", "skew"],
        # "Fare": ["min", "max", "median", "mean"],
    }
)



# 分组统计 split-apply-combine模式
df[["Sex", "Age"]].groupby("Sex").mean()

titanic.groupby(["Sex", "Pclass"])["Fare"].mean()



# titanic.sort_values

titanic.sort_values(by="Age").head()
titanic.sort_values(by=['Pclass', 'Age'], ascending=False).head()


我希望三个站点的值是彼此相隔的单独列。  数据透视表  ___.>


no2_subset.pivot(columns="location", values="value")





df

import copy

pd.concat([df,df2],axis=1)


### 2. 数据合并与连接

#### 合并（Merge）




### 3. 数据透视表与交叉表

#### 数据透视表（Pivot Table）



# 创建DataFrame
data = {'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'],
        'B': ['one', 'one', 'two', 'two', 'one', 'one'],
        'C': ['small', 'large', 'large', 'small', 'small', 'large'],
        'D': [1, 2, 2, 3, 3, 4]}
df = pd.DataFrame(data)

# 创建数据透视表
pivot_table = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'], aggfunc='sum')
print(pivot_table)



#### 交叉表（Crosstab）



# 创建交叉表
crosstab = pd.crosstab(df['A'], df['C'])
print(crosstab)



### 4. 重塑与透视

#### 重塑（Reshape）



# 创建DataFrame
data = {'A': ['foo', 'bar', 'baz'], 'B': [1, 2, 3], 'C': [4, 5, 6]}
df = pd.DataFrame(data)

# 使用melt函数
melted = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
print(melted)



#### 透视（Pivot）



# 创建DataFrame
data = {'A': ['foo', 'foo', 'bar', 'bar'], 'B': ['one', 'two', 'one', 'two'], 'C': [1, 2, 3, 4]}
df = pd.DataFrame(data)

# 使用pivot方法
pivoted = df.pivot(index='A', columns='B', values='C')
print(pivoted)



### 5. 分组与聚合

#### 分组操作（GroupBy）



# 创建DataFrame
data = {'A': ['foo', 'bar', 'foo', 'bar'], 'B': ['one', 'one', 'two', 'two'], 'C': [1, 2, 3, 4]}
df = pd.DataFrame(data)

# 分组并聚合
grouped = df.groupby('A').sum()
print(grouped)

# 多层分组与多重聚合
multi_grouped = df.groupby(['A', 'B']).agg({'C': ['sum', 'mean']})
print(multi_grouped)



### 6. 时间序列数据处理

#### 时间序列数据



# 创建时间序列DataFrame
date_rng = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df['data'] = pd.Series(range(1, len(df)+1))

# 设置日期时间索引
df.set_index('date', inplace=True)
print(df)

# 时间重采样
resampled = df.resample('2D').sum()
print(resampled)



### 7. 高级数据处理技巧

#### 条件合并



import numpy as np

# 创建两个DataFrame
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

# 条件合并
merged = pd.merge(df1, df2, on='key', how='outer')
merged['value'] = np.where(pd.notnull(merged['value1']), merged['value1'], merged['value2'])
print(merged)



#### 自定义函数与应用



# 创建DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
df = pd.DataFrame(data)

# 使用apply方法
df['C'] = df['A'].apply(lambda x: x * 2)
print(df)

# 自定义聚合函数
def custom_agg(x):
    return x.max() - x.min()

grouped = df.groupby('A').agg(custom_agg)
print(grouped)



### 8. 性能优化

#### 性能调优



import dask.dataframe as dd

# 创建大数据集
large_df = pd.DataFrame({'A': range(1000000), 'B': range(1000000)})

# 使用dask进行大数据处理
dask_df = dd.from_pandas(large_df, npartitions=10)
result = dask_df.groupby('A').sum().compute()
print(result)



### 9. 实战案例与项目

#### 实战案例



# 示例：多表数据的实际业务场景应用
# 假设有两个表：订单表和客户表

# 创建订单表
orders = pd.DataFrame({
    'order_id': [1, 2, 3, 4],
    'customer_id': [1, 2, 1, 3],
    'amount': [100, 200, 150, 300]
})

# 创建客户表
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'customer_name': ['Alice', 'Bob', 'Charlie']
})

# 合并订单表和客户表
merged_data = pd.merge(orders, customers, on='customer_id')
print(merged_data)

# 分析每个客户的总订单金额
customer_total = merged_data.groupby('customer_name')['amount'].sum().reset_index()
print(customer_total)






import dask.dataframe as dd

import pandas as pd
import numpy as np
import dask.dataframe as dd



# 数据透视表与交叉表
def create_pivot_table(df, values, index, columns, aggfunc='sum'):
    """
    创建数据透视表
    :param df: DataFrame
    :param values: 透视表值
    :param index: 行索引
    :param columns: 列索引
    :param aggfunc: 聚合函数
    :return: 数据透视表
    """
    return pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=aggfunc)

def create_crosstab(df, row, col):
    """
    创建交叉表
    :param df: DataFrame
    :param row: 行变量
    :param col: 列变量
    :return: 交叉表
    """
    return pd.crosstab(df[row], df[col])

# 重塑与透视
def reshape_dataframe(df, id_vars, value_vars):
    """
    使用melt函数重塑DataFrame
    :param df: DataFrame
    :param id_vars: 保持不变的列
    :param value_vars: 需要重塑的列
    :return: 重塑后的DataFrame
    """
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars)

def pivot_dataframe(df, index, columns, values):
    """
    使用pivot方法透视DataFrame
    :param df: DataFrame
    :param index: 行索引
    :param columns: 列索引
    :param values: 透视值
    :return: 透视后的DataFrame
    """
    return df.pivot(index=index, columns=columns, values=values)

# 分组与聚合
def group_and_aggregate(df, by, aggfunc='sum'):
    """
    分组并聚合DataFrame
    :param df: DataFrame
    :param by: 分组键
    :param aggfunc: 聚合函数
    :return: 聚合后的DataFrame
    """
    return df.groupby(by).agg(aggfunc)

def multi_group_and_aggregate(df, by, agg_dict):
    """
    多层分组与多重聚合
    :param df: DataFrame
    :param by: 分组键
    :param agg_dict: 聚合字典
    :return: 聚合后的DataFrame
    """
    return df.groupby(by).agg(agg_dict)

# 时间序列数据处理
def create_time_series(start, end, freq='D'):
    """
    创建时间序列DataFrame
    :param start: 开始日期
    :param end: 结束日期
    :param freq: 频率
    :return: 时间序列DataFrame
    """
    date_rng = pd.date_range(start=start, end=end, freq=freq)
    df = pd.DataFrame(date_rng, columns=['date'])
    df['data'] = pd.Series(range(1, len(df)+1))
    df.set_index('date', inplace=True)
    return df

def resample_time_series(df, rule, aggfunc='sum'):
    """
    时间重采样
    :param df: 时间序列DataFrame
    :param rule: 重采样规则
    :param aggfunc: 聚合函数
    :return: 重采样后的DataFrame
    """
    return df.resample(rule).agg(aggfunc)

# 高级数据处理技巧
def conditional_merge(df1, df2, key, condition_column, new_column):
    """
    条件合并两个DataFrame
    :param df1: 第一个DataFrame
    :param df2: 第二个DataFrame
    :param key: 合并键
    :param condition_column: 条件列
    :param new_column: 新列名
    :return: 合并后的DataFrame
    """
    merged = pd.merge(df1, df2, on=key, how='outer')
    merged[new_column] = np.where(pd.notnull(merged[condition_column[0]]), merged[condition_column[0]], merged[condition_column[1]])
    return merged

def apply_custom_function(df, column, func):
    """
    使用自定义函数处理DataFrame
    :param df: DataFrame
    :param column: 需要处理的列
    :param func: 自定义函数
    :return: 处理后的DataFrame
    """
    df[column] = df[column].apply(func)
    return df

def custom_aggregate(df, by, func):
    """
    自定义聚合函数
    :param df: DataFrame
    :param by: 分组键
    :param func: 自定义聚合函数
    :return: 聚合后的DataFrame
    """
    return df.groupby(by).agg(func)

# 性能优化
def optimize_performance(df, npartitions=10):
    """
    使用Dask进行大数据处理
    :param df: 大数据集DataFrame
    :param npartitions: 分区数
    :return: 处理后的DataFrame
    """
    dask_df = dd.from_pandas(df, npartitions=npartitions)
    return dask_df.groupby('A').sum().compute()

# 实战案例与项目
def merge_orders_and_customers(orders, customers):
    """
    合并订单表和客户表
    :param orders: 订单表DataFrame
    :param customers: 客户表DataFrame
    :return: 合并后的DataFrame
    """
    return pd.merge(orders, customers, on='customer_id')

def analyze_customer_total(merged_data):
    """
    分析每个客户的总订单金额
    :param merged_data: 合并后的DataFrame
    :return: 每个客户的总订单金额DataFrame
    """
    return merged_data.groupby('customer_name')['amount'].sum().reset_index()






"""
数据分析工具库
用于常用的数据分析
antuor:LSing
datetime:2021-09-19 20:38:05
"""
import os
import pandas as pd
from ..basis.toolbasis import BasisTools


def normalization(df, feature):
    """
    数据标准化
    :param df:
    :param feature:
    :return:
    """
    series = df[feature]
    return (series - series.mean()) / (series.std())


def isnull(df):
    """
    判断为空
    :param df:
    :return:
    """
    # pd.isnull(train).values.any()
    return pd.isnull(df)


def skew(df, feature):
    #
    # 基本上，偏度度量了实值随机变量的均值分布的不对称性。让我们计算损失的偏度：
    # stats.mstats.skew(train['loss']).data
    # 对数据进行对数变换通常可以改善倾斜，可以使用 np.log
    # stats.mstats.skew(np.log(train['loss'])).data
    # 连续值特征
    # train[cont_features].hist(bins=50, figsize=(16,12))
    return df[feature].skew


def fill_null(df, feature, type='', fillnum=0):
    """
    填充缺失值
    :param df:
    :param feature:
    :param type:
    :return:
    """
    if type == 'fill_mean':
        df[feature] = df[feature].fillna(df[feature].mean())
    elif type == 'fill_number':
        df[feature] = df[feature].fillna(fillnum)
    elif type == 'fill_newtype':
        df[feature] = df[feature].fillna('Null')


def divide_box_operation(self, df, feature, divide_point=[10, 20, 30]):
    """
    分箱操作
    :param df:
    :param feature:
    :param divide_point:
    :return:
    """
    return pd.cut(df[feature], bins=divide_point)


#
# # 对于额外使用流量进行分箱处理
# bin2=[-2500,-2000,-1000,-500,0,500,1000,2500]
# data['flow_label']=pd.cut(data.extra_flow,bins=bin2)
# data.head()
#
# # 对于额外通话时长进行分箱处理
# bin1=[-3000,-2000,-500,0,500,1000,2000,3000,5000]
# data['time_label']=pd.cut(data.extra_time,bins=bin1)
# data.head()
#

def counts_value(self):
    counts_value = []
    for item in self.columns:
        counts_value.append(self.dataFrame[item].value_counts)
    return counts_value


def add_new_feature(df1, df2, feature_on):
    """
    将两个数据融合在一起
    :param df1: 主要dataframe
    :param df2: 被加dataframe
    :param feature_on: 按照此特征为标准进行
    :return: 返回加好的数据
    """
    return df1.merge(df2, left_on=feature_on, right_on=feature_on, how='left')


def concat_data(paths_list):
    """
    将同一文件中的csv读成DF并合并在一起
    :param file_path:
    :return:
    """
    DF = pd.DataFrame()
    for paths in paths_list:
        data = pd.read_csv(paths)
        DF = DF.append(data)
    return DF


def plot(df, kind='bar', fontsize=15):
    df.plot(kind=kind, fontsize=fontsize)


def nunique(df, feature):
    return df[feature].nunique()


class Descruptive_analysis_tools(BasisTools):
    def __init__(self, df):
        super(Descruptive_analysis_tools, self).__init__()
        self.df = df

    def info(self):
        return self.df.info()

    def describe(self):
        return self.df.describe()

    def head(self, num):
        return self.df.head(num)

    def tail(self, num):
        return self.df.tail(num)

    def sample(self, num):
        return self.df.sample(num)

    def count(self, df):
        '''统计数量'''
        df.value_counts()

    def select_dtypes(self):
        return self.df.select_dtypes('object').describe(), \
               self.df.select_dtypes('float').describe()

    def get_dtype(self, feature):
        return self.df.dtypes[feature]


class Exploratory_analysis_tools(BasisTools):
    def __init__(self, df):
        """
        探索性分析工具
        :param df:
        """
        super(Exploratory_analysis_tools, self).__init__()
        self.df = df

    def corrmat_(self, feature, method='pearson'):
        """
        相关性分析
           # pearson：相关系数来衡量两个数据集合是否在一条线上面，即针对线性数据的相关系数计算，针对非线性数据便会有误差。
           # spearman：非线性的，非正太分析的数据的相关系数
           # kendall：用于反映分类变量相关性的指标，即针对无序序列的相关系数，非正太分布的数据
           # 上面的结果验证了，pearson对线性的预测较好，对于幂函数，预测差强人意。

        :param feature: 特征
        :param method: 方式
        :return: 相关矩阵
        """
        return self.df[feature].corr(method=method)







class Data_dealer(BasisTools):
    def __init__(self):
        super(Data_dealer, self).__init__()
        """"""

    def groupby_agg(self, df, group: list, agg: list):
        """
        分组聚合
        :param df:
        :param group: list
        :param agg: list
        :return:
        """
        return df.groupby(group).agg(agg)

    def groupby_apply(self, df, axis):
        """
        计算总和
        :return:
        """
        return df.apply(lambda x: x.sum(), axis=axis)

    def drop_dup(self, df, label):
        """
        去重
        :param df:
        :return:
        """
        return df.drop_duplicates(label)

    def reset_index(self, df):
        df.index = range(len(df))

    def df2list(self, df, feature):
        return df[feature].tolist()

    def read_necessary(self, unnecess=[]):
        """
        读取csv 去除不需要的字段
        :param path: 路径
        :return: DataFrame
        """
        data = pd.read_csv(self.path)
        for name in unnecess:
            try:
                data = data.drop(name, axis=1)
            except Exception as e:
                print(name, ':', e)
        return data

    def save_csv(self,pd,save_path):
        pd.to_csv(save_path,index=None)

    def set_option(self):
        """
        取消叠行列
        :return:
        """
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)

    def df_is_emply(self,df):
        if df.empty:
            return True
        else:
            return False

    def df_safe_read(self, path):
        if not os.path.exists(path):
            return pd.DataFrame()
        else:
            return pd.read_csv(path)



## notebook

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import io

# 创建一个新的Notebook对象
nb = nbf.v4.new_notebook()

# 添加Markdown单元格
markdown_cell = nbf.v4.new_markdown_cell("# 这是一个Markdown单元格\n\n以下是一些关于如何写一个好的Prompt的指南。")
nb.cells.append(markdown_cell)

# 添加代码单元格
code_cell = nbf.v4.new_code_cell("""# 示例Prompt
prompt = \"\"\"
请撰写一份关于人工智能未来发展的详细报告。报告应包括以下几个方面：
1. 人工智能的历史背景。
2. 当前的主要应用领域。
3. 未来五年的发展趋势预测。
4. 人工智能可能带来的社会和经济影响。
5. 相关的伦理和法律问题。

报告应使用学术写作风格，长度为3000字左右，并引用最新的研究和数据。
\"\"\"
print(prompt)
""")
nb.cells.append(code_cell)
nb.cells.append(nbf.v4.new_code_cell('sdfsdf'))

# 保存为一个新的ipynb文件
notebook_filename = 'example_notebook.ipynb'
with open(notebook_filename, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

# 运行Notebook并捕获输出和错误信息
def run_notebook(notebook_filename):
    global xx
    with open(notebook_filename) as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': './'}})
    except Exception as e:
        xx = e
        print("Error during execution:", e)

    # 捕获输出和错误信息
    output_stream = io.StringIO()
    for cell in nb.cells:
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'stream':
                    output_stream.write(output.text)
                elif output.output_type == 'error':
                    output_stream.write('\n'.join(output.traceback))

    return output_stream.getvalue()

output = run_notebook(notebook_filename)
print(output)


import os
import nbformat

def search_notebooks(directory, keyword):
    result = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                with open(notebook_path, 'r', encoding='utf-8') as f:
                    notebook = nbformat.read(f, as_version=4)
                    for cell in notebook.cells:
                        if cell.cell_type == 'markdown' or cell.cell_type == 'code':
                            if keyword.lower() in cell.source.lower():
                                result.append((notebook_path, cell.source))
    return result

directory_to_search = './'  # 修改为你想搜索的目录
keyword_to_search = 'DD_TOKEN'  # 修改为你要搜索的关键词

results = search_notebooks(directory_to_search, keyword_to_search)

for notebook_path, cell_content in results:
    print(f"Found in {notebook_path}:\n{cell_content}\n{'-'*80}\n")




import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [24, 27, 22, 32],
    'Score': [85, 92, 78, 88]
}

df = pd.DataFrame(data)



styled_df = df.style.set_properties(**{
    'background-color': 'lightyellow',
    'color': 'black',
    'border-color': 'black'
})

def color_score(val):
    color = 'green' if val > 90 else 'red'
    return f'background-color: {color}'

styled_df = df.style.applymap(color_score, subset=['Score'])

styled_df = df.style.highlight_max(subset=['Score'], color='lightgreen')
styled_df = styled_df.highlight_min(subset=['Score'], color='lightcoral')

styled_df = df.style.set_table_styles(
    [{'selector': 'tr:hover',
      'props': [('background-color', 'yellow')]}]
)

styled_df

html = styled_df.render()
with open('styled_df.html', 'w') as f:
    f.write(html)







# 处理文本

import pandas as pd

data = {
    'Name': ['Alice,1', 'Bob,2', 'Charlie,3', 'David,4'],
    'Age': [24, 27, 22, 32],
    'Score': [85, 92, 78, 88]
}

df = pd.DataFrame(data)

df.Age  #具备点语法















data.corr()

data.std()

data.nunique()#返回唯一值的个数

data[1].unique()#返回列的所有唯一值 Series

data.skew()

data.kurt()

## 查看属性

data.shape,data.empty,data.dtypes,data[1].dtypes

## 处理缺失值



## 分箱

pd.cut(df[feature], bins=divide_point)




## 合并
![请添加图片描述](https://img-blog.csdnimg.cn/cf452565043e48138884358e68c1ad68.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBA5YKy6Zuq5a2k6bi_,size_17,color_FFFFFF,t_70,g_se,x_16)


## 数据合并
```python
df1.append(df2)：将df2中的行添加到df1的尾部
df.concat([df1, df2],axis=1)：将df2中的列添加到df1的尾部
df1.join(df2,on=col1,how='inner')：对df1的列和df2的列执行SQL形式的join

s=pd.merge(pd.DataFrame(sales['Quant'].isnull()),
           pd.DataFrame(sales['Val'].isnull()),
           left_index=True,
           right_index=True)

```
[数据合并详解](https://blog.csdn.net/lemonbit/article/details/108656194)

数据堆放
制作一个列表 让后按列放到现有df后面

```python
df_label['close'] = list(stock_data['close'][:len(df_label)])

def add_column(df,label_name,df_column):
    """
    for example: stock_data['close']
    """
    df[label_name] = list(df_column[:len(df)])
add_column(df_label,'close',stock_data['close'])
```


分组聚合

分组聚合

```python
df[df[col] > 0.5]：选择col列的值大于0.5的行
df.sort_values(col1)：按照列col1排序数据，默认升序排列
df.sort_values(col2, ascending=False)：按照列col1降序排列数据
df.sort_values([col1,col2], ascending=[True,False])：先按列col1升序排列，后按col2降序排列数据
df.groupby(col)：返回一个按列col进行分组的Groupby对象
df.groupby([col1,col2])：返回一个按多列进行分组的Groupby对象
df.groupby(col1)[col2]：返回按列col1进行分组后，列col2的均值
df.pivot_table(index=col1, values=[col2,col3], aggfunc=max)：创建一个按列col1进行分组，并计算col2和col3的最大值的数据透视表
df.groupby(col1).agg(np.mean)：返回按列col1分组的所有列的均值
data.apply(np.mean)：对DataFrame中的每一列应用函数np.mean
data.apply(np.max,axis=1)：对DataFrame中的每一行应用函数np.max
```
Q1=sales['Uprice'].quantile(0.25)
Q3=sales['Uprice'].quantile(0.75)
IQR = Q3-Q1
outer = sales['out'].groupby(sales['Prod']).sum().sort_values(ascending=False)
产品销售数量分析
group = sales.groupby('Prod')
upp = group.sum()['Quant'].sort_values(ascending=False)

```





## 画图

df1.merge(df2, left_on=feature_on, right_on=feature_on, how='left')
def plot(df, kind='bar', fontsize=15):
    df.plot(kind=kind, fontsize=fontsize)


```python
inOutFlag1.plot.bar()

data2 = data.drop(['alt','x','y'],axis=1)
data2.plot()


保存图片
ax=df.plot()
fig=ax.get_figure()
fig.savefig(r'filepath\name.png')

```

data.drop_duplicates(label)

df.apply(lambda x: x.sum(), axis=axis)
df.groupby(group).agg(agg)
df[feature].corr(method=method)


def counts_value(self):
    counts_value = []
    for item in self.columns:
        counts_value.append(self.dataFrame[item].value_counts)
    return counts_value


def fill_null(df, feature, type='', fillnum=0):
    """
    填充缺失值
    :param df:
    :param feature:
    :param type:
    :return:
    """
    if type == 'fill_mean':
        df[feature] = df[feature].fillna(df[feature].mean())
    elif type == 'fill_number':
        df[feature] = df[feature].fillna(fillnum)
    elif type == 'fill_newtype':
        df[feature] = df[feature].fillna('Null')



df[feature].skew

pd.isnull(df)

函数pandas.DataFrame.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index= False)主要用来去除重复项，返回

subset：默认为None
去除重复项时要考虑的标签，当subset=None时所有标签都相同才认为是重复项


keep： {‘first’, ‘last’, False}，默认为‘first’
keep=‘first’时保留该标签第一次出现时的样本，之后重复出现的全部丢弃。
keep=‘last’表示保留该标签最后一次出现的样本，
keep=False时该标签重复的样本全部丢弃

inplace:bool，默认为False
inplace=False时返回去除重复项后的DataFrame,原来的DataFrame不改变。
inplace=True时返回空值，原来DataFrame被改变

ignore_index:bool，默认为False
ignore_index=False时，丢弃重复值之后的DataFrame的index不改变
ignore_index=True时，丢弃重复值之后的DataFrame的index重新变为0, 1, …, n-1


df_new =  df.reset_index(drop=True)


df3 = df1.append(df2)



数据选取
```python
df[col]：根据列名，并以Series的形式返回列
df[[col1, col2]]：以DataFrame形式返回多列
s.iloc[0]：按位置选取数据
s.loc['index_one']：按索引选取数据
df.iloc[0,:]：返回第一行
df.iloc[0,0]：返回第一列的第一个元素
df.values[:,:-1]:返回除了最后一列的其他列的所以数据
df.query('[1, 2] not in c'): 返回c列中不包含1，2的其他数据集
```

数据清理
```python
df.columns = ['a','b','c']：重命名列名
pd.isnull()：检查DataFrame对象中的空值，并返回一个Boolean数组
pd.notnull()：检查DataFrame对象中的非空值，并返回一个Boolean数组

sum(sales['Quant'].isnull()) 缺失值统计
df.dropna()：删除所有包含空值的行
df.dropna(axis=1)：删除所有包含空值的列
df.dropna(axis=1,thresh=n)：删除所有小于n个非空值的行
df.fillna(x)：用x替换DataFrame对象中所有的空值
s.astype(float)：将Series中的数据类型更改为float类型
s.replace(1,'one')：用‘one’代替所有等于1的值
s.replace([1,3],['one','three'])：用'one'代替1，用'three'代替3
df.rename(columns=lambda x: x + 1)：批量更改列名
df.rename(columns={'old_name': 'new_ name'})：选择性更改列名
df.set_index('column_one')：更改索引列
df.rename(index=lambda x: x + 1)：批量重命名索引




totS.hist(bins=200)
totP.hist(bins=200)


x=dict(list(q.groupby('instrument',as_index=True)))
将分组后的内容转换为字典



# @title numpy


import numpy as np







# pandas numpy torch 的不同

import numpy as np
import pandas as pd
import torch

## 创建

# 创建一个demo 服从标准正态分布
np.random.randn(3,4,5)
pd.DataFrame(np.random.randn(3,4,5))
torch.randn(3,4,5)

# 创建一个demo 服从自定义正态分布 离散正态
np.random.normal(1,2,(3,4))
pd.DataFrame(np.random.normal(1,2,(3,4)))
torch.normal(1,2,(3,4))


# 创建一个demo 服从均匀分布
np.random.rand(3,2)
pd.DataFrame(np.random.rand(3,2))
torch.rand(3,2)

# 在[1, 10)之间离散均匀抽样，数组形状为3行2列
np.random.randint(1, 10, (3, 2),dtype=np.int64)
pd.DataFrame(np.random.randint(1, 10, (3, 2),dtype=np.int64))
torch.randint(1,10,(3,2))

# 创建一个空值
# N 维数组对象 ndarray是类,而array是函数 本质上来说就是array对象
np.array()
pd.DataFrame()
torch.Tensor()

#创建
ary = np.arange(1, 10) #range升级版
ary = np.zeros(10, dtype='int32')
ary = np.ones((2, 3), dtype='float32')
ary = np.array([1, 2, 3, 4, 5], ndmin =  2)


# Concat

a = np.random.randn(3,4,5)
b = np.random.randn(3,4,5)
c = np.concatenate((a,b),axis=0)
c.shape

np.append  np.extend
np.stack()
np.hstack()
np.vstack()
np.dstack()

a = pd.DataFrame(np.random.randn(3,5))# pandas 只处理2维数据
b = pd.DataFrame(np.random.randn(3,5))
c = pd.concat([a,b],axis=0)
c.shape

pandas join() merge() concat()

a = torch.randn(3,4,5)
b = torch.randn(3,4,5)
c = torch.cat([a,b],dim=0)
c.shape

# 循环拼接

a = np.random.randn(3,6)
for i in range(19):
    b = np.random.randn(3,6)
    a = np.concatenate((a,b),axis=0)
a.shape,b.shape

a = pd.DataFrame()
for i in range(20):
    b = pd.DataFrame(np.random.randn(3,6))
    a = pd.concat([a,b],axis=0)
a.shape,b.shape

a = torch.Tensor()
for i in range(20):
    b = torch.randn(3,4,5)
    a = torch.cat([a,b],dim=0)
a.shape,b.shape

torch.Tensor(np.random.randn(3,6))

# 互相转换

# Tensor()-> numpy
tran_normalized.cpu().detach().numpy()
#numpy -> Tensor()
torch.Tensor(np)

# pandas->numpy
Df.values
# numpy-> pandas
pd.DataFrame(num)

#pandas<->Tensor
# 间接转换


# reshape

np.array([i for i in range(60)]).reshape(3,4,5)
pd.DataFrame([i for i in range(60)])#pandas没有
torch.Tensor([i for i in range(60)]).reshape(3,4,5)


# 属性

np 属性
- shape - 维度
- dtype - 元素类型 #pandas无
- size - 元素数量 #tensor无 可以用size()
- ndim - 维数，len(shape)
- T - 数组对象的转置视图

a = np.array([i for i in range(60)]).reshape(3,4,5)
b = np.array([i for i in range(60)]).reshape(3,4,5)
a+b,a-b,a*b,a+3,a>3

pd.DataFrame([i for i in range(60)])#pandas没有
torch.Tensor([i for i in range(60)]).reshape(3,4,5)








## numpy 切片
```python
# 一维
import numpy as np
a=np.arange(10)
print(a[0:9])  # 包头不包尾
print(a[3:6])
print(a[:5])  # :前面不写就是从下标为0开始
print(a[5:])  # :后面不写就是一直到最后一个元素
print(a[:])   # :前后都不写就是从头到尾

# 二维
b= np.mat(np.arange(20).reshape(4,5))
print(b[1:3,2:5])   # 先取第一维中下标为1,2的2部分，再取第二维中下标为2,3,4这3部分
print(b[:2,2:])     # 同理，前面不写从头开始，后面不写一直到末尾
print(b[:2,3])      # 当然，也可以在某维度上只取一行

# 三维
c= np.arange(60).reshape(3,4,5)
print(c)
print(c[:2,2:4,1:4])  # 从外向内一层一层的割，切割不改变矩阵维度
```


import pandas as pd

def ask_bid(x):
    data = x
    return data
dataask = data_all['ask_price'].apply(ask_bid)
dataask

data_all = pd.read_csv('sz000001.csv',index_col=0)
data_all










numpy 笔记

创建numpy数组

import numpy as np
a = np.array(range(12))
b = np.random.randn(3,3)

查看属性

a.shape

查看统计量

改变维度

a1 = a.reshape(3,4)

a2 = a.ravel()# 将数组撵平
a3 = a.flatten()# 展平

# 直接改变原来数组
a.shape = (6,2)
a.resize(3,4)
a

改变数值

计算

合并数组


一维数组的组合方案
a = np.arange(1, 9)
b = np.arange(9, 17)
print(a)
print(b)
print(np.row_stack((a, b)))  # 形成两行
print(np.column_stack((a, b)))  # 形成两列
```

```python


np.average(
    $var$,   #求平均的数组
    $var$,   #权重
)


# 裁剪
# 将调用数组中小于和大于下限和上限的元素替换为下限和上限，返回裁剪后的数组，调
# 用数组保持不变。
np.clip($var$,   #需要裁剪的数组
        $var$,   #最小值
        $var$,   #最大值
)

# 压缩 (只保留a>5的元素)
"""mask = np.all([a > 3, a < 7], axis=0)
print(a.compress(mask))
print(a[mask])"""

# 返回由调用数组中满足条件的元素组成的新数组。

a

mask = np.all([a > 1, a < 3], axis=0)
a.compress(mask)

## 计算(关系型)

### 求均值

a.mean(axis=0),a.mean(axis=1),a.mean(),np.mean(a,axis=0),np.mean(a,axis=1),np.mean(a)

### 求和

a.sum(axis=0),a.sum(axis=1),a.sum(),np.sum(a,axis=0),np.sum(a,axis=1),np.sum(a)

### 标准差

a.std(axis=0),a.std(axis=1),a.std(),np.std(a,axis=0),np.std(a,axis=1),np.std(a)

### 乘法

a.prod(axis=0),a.prod(axis=1),a.prod(),np.prod(a,axis=0),np.prod(a,axis=1),np.prod(a)

### 累乘

a.cumprod(axis=0),a.cumprod(axis=1),a.cumprod(),np.cumprod(a,axis=0),np.cumprod(a,axis=1),np.cumprod(a)

### 累加

a.cumsum(axis=0),a.cumsum(axis=1),a.cumsum(),np.cumsum(a,axis=0),np.cumsum(a,axis=1),np.cumsum(a)

### 计算(非关系型)

### 开平方

np.sqrt(a)

### 矩阵间计算

np.outer([10, 20, 30], a)   # 外积'''








# matplotlib笔记

!/usr/share/fonts/ #将字体放到指定目录下
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False# 步骤二（解决坐标轴负数的负号显示问题）













plt.plot([extent[0], extent[1],extent[1], extent[0],extent[0]],
         [extent[2], extent[2],extent[3], extent[3],extent[2]], c=c)


plt.show()

plt.imshow(image, extent=extent)


plt.savefig(title)
plt.show()
```




```python
# 消掉打印不完全中间的省略号

pd.set_option('display.max_columns', 1000)

pd.set_option('display.width', 1000)

pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_colwidth', None)  # 设置列宽为无限制

pd.set_option('display.max_rows', None)

pd.set_option('display.expand_frame_repr', False)  # 设置不折叠数据
```





数据分析工具    高级版/ GPU版 Dask

DataFrame和Series 是pandas中的两个核心概念
DataFrame 可以看做是一个列向量组成的向量组
Series 可以看做这个列向量
列向量有基本的属性: 列名和元素, 元素应符合基本的性质

```python
df = pd.DataFrame(
{
"Name": [
	"Braund, Mr. Owen Harris",
	"Allen, Mr. William Henry",
	"Bonnell, Miss. Elizabeth",
	],
"Age": [22, 35, 58],
"Sex": ["male", "male", "female"],
}
)
```
DataFrame 本质是一个列向量的字典

```python
ages = pd.Series([22, 35, 58], name="Age")
pd.Series(data=[3,-5,'b',4],index=['a','b','c','d'])
```
Series 是列向量

```python
ages.describe()
ages.info()
ages.value_counts()
ages.dtypes
ages.head(8)
ages.tail(8)
ages.shape
ages.plot()
ages.plot.box()
ages.mean()
ages.median()
ages.count()
ages.sample()

# 文字处理
ages.str.lower()
ages.str.split(",")
ages.str.split(",").str.get(0)
ages.str.len()
ages.str.len().idxmax()
ages.str.replace('Da','cc')

```
Series 的这些方法, DataFrame也具备, 本质上是对Series的遍历调用和统一展示
此外,还可以看看[[对Series的条件判断怎么做]]


关于时间序列的事
Timestamp
DatetimeIndex

---





######### pandaAI




构建DataLayer
```python
import pandasai as pai
df2 = pai.read_csv("test.csv")

# Create the data layer
companies = pai.create(
	path="my-org/companies2",# 存放路径
	df=df2,
	description="Customer companies dataset", # 对于数据集的描述
	# 定义数据集结构
	columns=[
		{
		"name": "company_name",
		"type": "string",
		"description": "The name of the company"
			},
		{
		"name": "revenue",
		"type": "float",
		"description": "The revenue of the company"
			},
		{
		"name": "region",
		"type": "string",
		"description": "The region of the company"
			}
		]
	)
```

> path 必须遵守以下结构 organization-identifier/dataset-identifier
> columns - type 类型如下:
> 		“string”: IDs, names, categories
> 		“integer”: counts, whole numbers
> 		“float”: prices, percentages
> 		“datetime”: timestamps, dates
> 		“boolean”: flags, true/false values



### 加载DataLayer
```python
# Load existing datasets
stocks = pai.load("my-org/companies")
companies = pai.load("my-org/companies2")

# Query using natural language
response = stocks.chat("What is the volatility of the Coca Cola stock?")
response = companies.chat("What is the average revenue by region?")

# Query using multiple datasets
result = pai.chat("Compare the revenue between Coca Cola and Apple", stocks, companies)
```


数据层结合SQL

```python

sql_table = pai.create(

# Format: "organization/dataset"

path="company/health-data",

  

# Optional description

description="Heart disease dataset from MySQL database",

  

# Define the source of the data, including connection details and

# table name

source={

"type": "mysql",

"connection": {

"host": "${DB_HOST}",

"port": 3306,

"user": "${DB_USER}",

"password": "${DB_PASSWORD}",

"database": "${DB_NAME}"

},

"table": "heart_data"

}

)
```




```python


# !pip install -q llama_index
# !pip install "pandasai>=3.0.0b2"
# !pip install pandasai-openai

import pandasai as pai
import yaml
import os
# import fire

def create_path(path):
    if os.path.exists(path):
        raise FileExistsError(f"路径已存在: {path}")
    else:
        os.makedirs(path)
        print(f"路径已创建: {path}")


def create_data_layer(data_path,csv_file,description,columns):
    # Create the data layer
    create_path(data_path)
    companies = pai.create(
        path=data_path,# 存放路径
        df=pai.read_csv(csv_file),
        description=description, # 对于数据集的描述
        #  定义数据集结构
        columns=columns
        )

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    config = load_config(config_path)
    data_layer_config = config.get("data_layer", {})

    name = data_layer_config.get("name")
    file_path = data_layer_config.get("file_path")
    description = data_layer_config.get("description")
    columns = data_layer_config.get("columns", [])

    create_data_layer(name, file_path, description, columns)


# if __name__ == "__main__":
#     fire.Fire(main)
```



构建Agent 并优化其效果

```python
# 要正确理解这里train 的含义, 其实是prompt层级的优化
import pandasai as pai
from pandasai import Agent

agent = Agent("data.csv")
agent.train(docs="The fiscal year starts in April")

response = agent.chat("What is the total sales for the fiscal year?")

print(response)

# The model will use the information provided in the training to generate a response
```

```python

from pandasai import Agent

agent = Agent("data.csv")  

# Train the model
query = "What is the total sales for the current fiscal year?"
# The following code is passed as a string to the response variable
response = '\n'.join([
'import pandas as pd',
'',
'df = dfs[0]',
'',
'# Calculate the total sales for the current fiscal year',
'total_sales = df[df[\'date\'] >= pd.to_datetime(\'today\').replace(month=4, day=1)][\'sales\'].sum()',
'result = { "type": "number", "value": total_sales }'
])
agent.train(queries=[query], codes=[response])
response = agent.chat("What is the total sales for the last fiscal year?")
print(response)
# The model will use the information provided in the training to generate a response

```





```python

!pip install -q llama_index
!pip install streamlit
!pip install "pandasai>=3.0.0b2"

# @title 基本结构

# @markdown 挂载网盘
from google.colab import drive
drive.mount('/content/drive')

# @markdown 配置信息
from google.colab import userdata
api_key = userdata.get('api_key')
api_base = userdata.get('api_base')

# @markdown 设置语言模型


# @markdown 构建Tools Agent RAGTools


# 创建RAG工具
txt_path = "drive/MyDrive/data/txt_demo"
from llama_index.core.tools import ToolMetadata

# 加载文档并创建知识库索引



# 创建查询引擎工具
query_engine_tool = QueryEngineToo


# PandasAI代码工具

# @title 导入和初始化

import pandasai as pai
# Get your API key from https://app.pandabi.ai
pai.api_key.set("PAI-e874a529-0404-4829-ae8c-543b25763088")

# @title 制作数据
import pandas as pd
df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
})
df.to_csv('test.csv')

df



# @title 使用read方式加载数据
df1 = pai.read_csv('test.csv')
df1.chat('计算平均幸福指数')

# @title 加载数据并对话
df = pai.DataFrame(df)
df.chat("Which are the 5 happiest countries? and Plot it")

!mkdir -p my-org/companies2

# @title 创建数据层 data layer
import pandasai as pai

# Load your data
df2 = pai.read_csv("test.csv")

# Create the data layer
companies = pai.create(
  path="my-org/companies2",# 存放路径
  df=df2,
  description="Customer companies dataset", # 对于数据集的描述
  #  定义数据集结构
  columns=[
    {
      "name": "company_name",
      "type": "string",
      "description": "The name of the company"
    },
    {
      "name": "revenue",
      "type": "float",
      "description": "The revenue of the company"
    },
    {
      "name": "region",
      "type": "string",
      "description": "The region of the company"
    }
  ]
)



## 数据层 Data Layer

# @title 加载data layer 并询问

# Load existing datasets
stocks = pai.load("my-org/companies")
companies = pai.load("my-org/companies2")

# Query using natural language
response = stocks.chat("What is the volatility of the Coca Cola stock?")
response = companies.chat("What is the average revenue by region?")

# Query using multiple datasets
result = pai.chat("Compare the revenue between Coca Cola and Apple", stocks, companies)


# @title Semantic Layer
# 提供局域增强和数据清洗能力
# 要使用Semantic layer 就要从创建schema开始


# @title new schema
# @markdown 参考  -> 创建数据层 data layer

# path 必须遵循 Must follow the format: “organization-identifier/dataset-identifier”
# 标识符遵循唯一性


# columns =
"""

type (str): Data type of the column
“string”: IDs, names, categories
“integer”: counts, whole numbers
“float”: prices, percentages
“datetime”: timestamps, dates
“boolean”: flags, true/false values
description (str): Clear explanation of what the column represents
​

"""

# @title SQL 数据集做 schema

sql_table = pai.create(
    # Format: "organization/dataset"
    path="company/health-data",

    # Optional description
    description="Heart disease dataset from MySQL database",

    # Define the source of the data, including connection details and
    # table name
    source={
        "type": "mysql",
        "connection": {
            "host": "${DB_HOST}",
            "port": 3306,
            "user": "${DB_USER}",
            "password": "${DB_PASSWORD}",
            "database": "${DB_NAME}"
        },
        "table": "heart_data"
    }
)


# @title YAML 配置方式
# https://docs.getpanda.ai/v3/semantic-layer/new
# 暂时没有需求


# @title NL Layer
# 自然语言层  强调自然语言处理能力
df.load("organization/dataset-name")
df.push()# 推送到数据平台


# Config NL Layer
import pandasai as pai

pai.config.set({
   "llm": "openai",
   "save_logs": True,
   "verbose": False,
   "max_retries": 3
})


## 使用视图

# @title Working with Views

# 其实指的是透视表
# 这例通过修改YAML 的配置文件来实现, 这里应该是实现多表联查的关键
#https://docs.getpanda.ai/v3/semantic-layer/views
# 暂无需求



## 新增数据源

# @title 很多数据源
# 主要增加一些SQL的数据源, 云服务的数据源
# https://docs.getpanda.ai/v3/data-ingestion
# 暂无需求




## 数据转换 Data Transformations

# @title 数据转换
# 理解是通过YAML的配置方式来改变数据的类型,
# https://docs.getpanda.ai/v3/transformations
# 暂无需求



## Chat 和output formats

# 聊天的输出格式
# 基础用法
import pandasai as pai
df_customers = pai.load("company/customers")
response = df_customers.chat("Which are our top 5 customers?")

# 多数据混合问询
import pandasai as pai
df_customers = pai.load("company/customers")
df_orders = pai.load("company/orders")
df_products = pai.load("company/products")

response = pai.chat('Who are our top 5 customers and what products do they buy most frequently?', df_customers, df_orders, df_products)

#
'''
DataFrame Response
Used when the result is a pandas DataFrame. This format preserves the tabular structure of your data and allows for further data manipulation.

​
Chart Response
Handles visualization outputs, supporting various types of charts and plots generated during data analysis.

​
String Response
Returns textual responses, explanations, and insights about your data in a readable format.

​
Number Response
Specialized format for numerical outputs, typically used for calculations, statistics, and metrics.

​
Error Response
Provides structured error information when something goes wrong during the analysis process.

​

'''



## Agent

You can train PandaAI to understand your data better

# 要正确理解这里train 的含义, 其实是prompt层级的优化

import pandasai as pai
from pandasai import Agent

pai.api_key.set("your-pai-api-key")

agent = Agent("data.csv")
agent.train(docs="The fiscal year starts in April")

response = agent.chat("What is the total sales for the fiscal year?")
print(response)
# The model will use the information provided in the training to generate a response


from pandasai import Agent

agent = Agent("data.csv")

# Train the model
query = "What is the total sales for the current fiscal year?"
# The following code is passed as a string to the response variable
response = '\n'.join([
    'import pandas as pd',
    '',
    'df = dfs[0]',
    '',
    '# Calculate the total sales for the current fiscal year',
    'total_sales = df[df[\'date\'] >= pd.to_datetime(\'today\').replace(month=4, day=1)][\'sales\'].sum()',
    'result = { "type": "number", "value": total_sales }'
])

agent.train(queries=[query], codes=[response])

response = agent.chat("What is the total sales for the last fiscal year?")
print(response)

# The model will use the information provided in the training to generate a response


# Training with local Vector stores
# 暂无需求


# 客制化Custom Head

import pandas as pd
import pandasai as pai

# Your original dataframe
df = pd.DataFrame({
    'sensitive_id': [1001, 1002, 1003, 1004, 1005],
    'amount': [150, 200, 300, 400, 500],
    'category': ['A', 'B', 'A', 'C', 'B']
})

# Create a custom head with anonymized data
head_df = pd.DataFrame({
    'sensitive_id': [1, 2, 3, 4, 5],
    'amount': [100, 200, 300, 400, 500],
    'category': ['A', 'B', 'C', 'A', 'B']
})

# Use the custom head
smart_df = pai.SmartDataframe(df, config={
    "custom_head": head_df
})


## 辅助性技术

# @title 向平台推送数据集

# https://docs.getpanda.ai/v3/getting-started  Sharing and collaboration
# 暂时没有需求


# 数据看板

# @title Command line interface

# https://docs.getpanda.ai/v3/cli  Command line interface
# 暂时没有需求


# @title Privacy & Security

# https://docs.getpanda.ai/v3/privacy-security#code-execution-and-sandbox-environment
# 暂时没有需求

# @title LLM 的设置
# you can configure it simply using pai.config.set(). Then, every time you use the .chat() method, it will use the configured LLM

# !pip install pandasai-openai
import pandasai as pai
from pandasai_openai import OpenAI

llm = OpenAI(api_token="my-openai-api-key")

# Set your OpenAI API key
pai.config.set({"llm": llm})

# 还有其他技术
# pip install pandasai-huggingface
# pip install pandasai-langchain

import pandasai as pai
from pandasai_langchain import LangchainLLM

llm = LangchainLLM(openai_api_key="my-openai-api-key")

pai.config.set({"llm": llm })

#pip install pandasai-local

import pandasai as pai
from pandasai_local import LocalLLM

ollama_llm = LocalLLM(api_base="http://localhost:11434/v1", model="codellama")

pai.config.set({"llm": ollama_llm})

#
import pandasai as pai
from pandasai_local import LocalLLM

lm_studio_llm = LocalLLM(api_base="http://localhost:1234/v1")

pai.config.set({"llm": lm_studio_llm })







# SmartDataFrame 是一个过去的方法
# SmartDatalake
# 暂无需求


smart_df = SmartDataframe(df, config={
    "llm": llm,                              # LLM instance
    "save_logs": True,                       # Save conversation logs
    "verbose": False                         # Print detailed logs
})

```



直接读取CSV文件
```python
df1 = pai.read_csv('test.csv')
df1.chat('计算平均幸福指数')
```

 包装DataFrame
```python
df = pai.DataFrame(df)
df.chat("Which are the 5 happiest countries? and Plot it")
```




```python

# !pip install -q llama_index
# !pip install "pandasai>=3.0.0b2"
# !pip install pandasai-openai

import pandasai as pai
import yaml
import os
import fire

def load_config(file_path):
    # 加载配置文件
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path):
    config = load_config(config_path)
    data_layer_config = config.get("data_layer", {})

    name = data_layer_config.get("name")
    file_path = data_layer_config.get("file_path")
    description = data_layer_config.get("description")
    columns = data_layer_config.get("columns", [])

    companies = pai.create(
        path=name,# 存放路径
        df=pai.read_csv(file_path),
        description=description, # 对于数据集的描述
        columns=columns # 定义数据集结构
        )


if __name__ == "__main__":
    fire.Fire(main)
```


```python
from pandasai_openai import OpenAI as OpenAI_pai
import pandasai as pai


def response_processor(response, png_file, df_file, verbose=False):
    """
    响应处理
    
    """
    if verbose:
        print(response.last_code_executed)
    if response.type == 'chart':
        response.save(png_file)
        return 'img saved'
    elif response.type == 'dataframe':
        response.value.to_csv(df_file)
        return 'df saved'
    elif response.type == 'string':
        return response.value
    elif response.type == 'number':
        return response.value
    else:
        return response


class DataAnalyseTool():
    def __init__(self, api_base, api_key, model="gpt-4o",
                 png_file='./temp.png', df_file='./temp.csv', verbose=False):
        self.llm = OpenAI_pai(api_base=api_base,
                              api_token=api_key,
                              model=model)
        self.data_layers = []
        pai.config.set({"llm": self.llm})
        self.png_file = png_file
        self.df_file = df_file
        self.verbose = verbose

    def load_data_layers(self, data_layers=[]):
        """
        加载数据层
        data_layers: 数据层
        """
        for data_layer in data_layers:
            try:
                sdf = pai.load(data_layer)
                self.data_layers.append(sdf)
            except Exception as e:
                print(f"Error loading data layer {data_layer}: {e}")

    def chat(self, query):
        response = pai.chat(query, *self.data_layers)

        # MaliciousQueryError : Query uses unauthorized table: users.

        return response_processor(response, png_file=self.png_file,
                                  df_file=self.df_file,
                                  verbose=self.verbose)
```


