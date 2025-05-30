```python

from tqdm.notebook import tqdm
from tqdm.asyncio import tqdm_asyncio

```

```python
from tqdm import tqdm
pbar = tqdm(带循环内容,total=100,
					 desc='Custom',
					 colour='blue')
pbar.update(1) # 更新进度条
pbar.close()
```

```python
for i in tqdm(range(100),total=200,desc='Custom', colour='blue'):

	print(i)
```




```python
import yaml

def load_yaml(file_path: str, encoding: str = 'utf-8'):	
	f = open(file_path, encoding=encoding)
	return yaml.load(f, Loader=yaml.FullLoader)

def save_yaml(v: dict, file_path: str):
	f = open(file_path, 'w')
	yaml.dump(v, f)
```

```python
# 数据查看 !pip install dtale



```bash
pip install python-dotenv
```

**创建 `.env` 文件**：在项目根目录下创建一个名为 `.env` 的文件，内容如下：
```
DATABASE_URL=postgres://user:password@localhost:5432/mydatabase
SECRET_KEY=mysecretkey
DEBUG=True
```


   ```python
   from dotenv import load_dotenv
   import os

   # 加载 .env 文件
   load_dotenv()

   # 访问环境变量
   database_url = os.getenv('DATABASE_URL')
   secret_key = os.getenv('SECRET_KEY')
   debug = os.getenv('DEBUG')

   print(f"Database URL: {database_url}")
   print(f"Secret Key: {secret_key}")
   print(f"Debug Mode: {debug}")
   ```







`contextlib` 是 Python 标准库中的一个模块，用于简化上下文管理器的创建和使用。上下文管理器是通过 `with` 语句使用的对象，它们在进入和退出代码块时执行一些设置和清理操作。`contextlib` 提供了几个有用的工具来创建和管理上下文管理器。

以下是 `contextlib` 的一些常见用法和示例：

### 1. 使用 `contextmanager` 装饰器
`contextmanager` 是 `contextlib` 模块中最常用的装饰器，用于将生成器函数转换为上下文管理器。

```python
from contextlib import contextmanager

@contextmanager
def my_context_manager():
    print("进入上下文")
    yield  # yield之前的代码相当于__enter__，yield之后的代码相当于__exit__
    print("退出上下文")

# 使用上下文管理器
with my_context_manager():
    print("在上下文中执行代码")
```

输出：
```
进入上下文
在上下文中执行代码
退出上下文
```

### 2. 使用 `closing` 管理资源
`closing` 是一个上下文管理器，用于确保某个对象在使用后正确关闭。它通常用于管理那些没有实现上下文管理器协议的对象。

```python
from contextlib import closing
import urllib.request

with closing(urllib.request.urlopen('http://www.python.org')) as page:
    for line in page:
        print(line.decode('utf-8'))
```

### 3. 使用 `suppress` 抑制异常
`suppress` 是一个上下文管理器，用于抑制指定的异常。

```python
from contextlib import suppress

with suppress(ValueError, TypeError):
    raise ValueError("这是一个ValueError")
print("异常被抑制，继续执行")
```

输出：
```
异常被抑制，继续执行
```

### 4. 使用 `redirect_stdout` 和 `redirect_stderr`
这两个上下文管理器用于重定向标准输出和标准错误。

```python
from contextlib import redirect_stdout
import sys

with open('output.txt', 'w') as f:
    with redirect_stdout(f):
        print("这将被写入output.txt文件")
```

### 5. 使用 `ExitStack` 管理多个上下文
`ExitStack` 是一个上下文管理器，用于管理多个嵌套的上下文管理器。

```python
from contextlib import ExitStack

with ExitStack() as stack:
    file1 = stack.enter_context(open('file1.txt', 'w'))
    file2 = stack.enter_context(open('file2.txt', 'w'))
    file1.write("写入file1.txt")
    file2.write("写入file2.txt")
```

### 6. 自定义上下文管理器
除了使用 `contextmanager` 装饰器，你还可以通过实现 `__enter__` 和 `__exit__` 方法来自定义上下文管理器。

```python
class MyContextManager:
    def __enter__(self):
        print("进入上下文")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("退出上下文")

    def do_something(self):
        print("在上下文中执行代码")

# 使用自定义上下文管理器
with MyContextManager() as cm:
    cm.do_something()
```

输出：
```
进入上下文
在上下文中执行代码
退出上下文
```

这些是 `contextlib` 模块的一些常见用法和示例。通过这些工具，你可以更方便地管理资源和处理异常，使代码更加简洁和健壮。


```python
import diskcache

cache = diskcache.Cache(cache_path)
# 存储
cache.set(key=key,value=value)
# 读取
result = cache.get(key)
```




编辑错误码
```python
class MonitorError(Exception):
	def __init__(self, message):
		self.message = message
		super().__init__(self.message)
```

抛出异常
```python
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError(f"Environment variable OPENAI_API_KEY is not set")
```

