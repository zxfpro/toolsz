
```python

import sys
import signal
from typing import Callable, Union
import functools


def external_handle_quit(quit_func: Union[Callable, None] = None):
    """
如何解决无法通过ctrl C 退出的问题

    是一个可以通过ctrl C 来进行退出的装饰器
    quit_func 表示传入Callable or None
    退出 模块
    default func:
        def quit(signum, frame):
            print('退出')
            sys.exit()

    except KeyboardInterrupt as e:
    退出函数无法解决 传入变量的事情 所以无法对东西做及时的保存
    """
    def outer_packing(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def quit(signum, frame):
                print('退出')
                sys.exit()
            _quit = quit_func or quit
            signal.signal(signal.SIGINT, _quit)
            signal.signal(signal.SIGTERM, _quit)
            result = func(*args, **kwargs)
            return result
        return wrapper
    return outer_packing

#other

```



```python
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
```


```

import json
import httpx
from mcp.server import FastMCP

app = FastMCP("text_work")

# Add an addition tool
@app.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


if __name__ == '__main__':
    app.run(transport='stdio')
```

```python
import sys
import traceback
try:
    # 可能引发异常的代码
    print(1 / 0)
except Exception as e:
    error_info = traceback.format_exc()
    print(error_info)

```



如果你希望类或对象只有特定属性，可以使用 __slots__ 来限制属性的添加

```python

class Student:
    __slots__ = ('name', 'age', 'score')
    def __init__(self, name, age):
        self.name = name
        self.age = age

```

```python
import re
class Infos:
    def __init__(self, key,value):
        self.key = key
        self.value = value
    
    def transform_key(self,key):
        # 使用正则表达式匹配并替换
        new_key = re.sub(r'(\w+)\[\d+\]', r'\1_list', key)
        return new_key

    def __eq__(self, other):
        # 自定义相等
        if isinstance(other, MyClass):
            key = self.transform_key(self.key)
            other_key = self.transform_key(other.key)

            return (key == other_key) and (type(self.value)==type(other.value))
        return False
    
    def __hash__(self):
        # 修改 set 中的值 {}
        return hash(self.transform_key(self.key) + str(type(self.value)))


```


```
!pip install locust
```

locustfile.py
```python
from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def test_endpoint(self):
        self.client.get("/your-endpoint")  # 替换为你的 FastAPI 服务端点

    @task
    def test_endpoint(self):
        payload = {
            "messages": [
                {
                    "content": "你好，你是谁?"
                }
            ]
        }
        
        self.client.post("/api/chat",json=payload)  # 替换为你的 FastAPI 服务端点

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)  # 每个用户等待 1 到 3 秒之间
```

```
!locust -f locustfile.py --host=http://localhost:8000
```

