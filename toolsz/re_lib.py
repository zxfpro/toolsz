""" 一些常用的正则工具"""
import re

def extract_python_code(text: str)->str:
    """从文本中提取python代码

    Args:
        text (str): 输入的文本。

    Returns:
        str: 提取出的python文本
    """
    pattern = r'```python([\s\S]*?)```'
    matches = re.findall(pattern, text)
    return matches

def extract_json_code(text:str)->str:
    """从文本中提取json代码

    Args:
        text (str): 输入的文本。

    Returns:
        str: 提取出的json文本
    """
    pattern = r'```json([\s\S]*?)```'
    matches = re.findall(pattern, text)
    return matches


"""

```python

+ 号代表前面的字符必须至少出现一次
* 号代表前面的字符可以不出现，也可以出现一次或者多次

```



```python
import re
code_match = re.search(r'### 3. Function\n\n```python(.*?)```',python_code,flags = re.DOTALL)
```

```python
import re
code_match = re.search(r'### 3. Function\n\n```python(.*?)```',python_code,flags = re.DOTALL)
```


```python
# re.findall
import re

text = "这个短语中有 2 个数字：1234 和 5678。"
pattern = r'\d+'

# 查找所有数字
matches = re.findall(pattern, text)
print(matches)

```

```python
# 搜索匹配  分组匹配

pattern = r'(\d{4})-(\d{2})-(\d{2})'  # 匹配日期格式（年-月-日）
text = "今天是 2023-10-05。"
matches = re.search(pattern, text)
if matches:
    year, month, day = matches.groups()
    print(f"年份: {year}, 月份: {month}, 日期: {day}")
```


```python
text = "Hello World 123"
pattern = r'^Hello'
match = re.match(pattern, text)
if match:
    print("匹配成功!", match.group())  # 输出: 匹配成功! Hello
```

```python
# 内容替换
# 使用 re.sub() 替换内容
text = "Hello 123 World 456"
cleaned_text = re.sub(r'\d+', '', text)  # 替换所有数字为空字符串
print(cleaned_text)  # 输出: "Hello  World  "

```


```python
%% 切分 %%
text = "Hello, World! How are you?"
split_result = re.split(r'\W+', text)
print(split_result)  # 输出: ['Hello', 'World', 'How', 'are', 'you']
```

```python
regex = re.compile(pattern)

# - **用途**：将正则表达式编译为正则表达式对象，便于多次使用。
regex = re.compile(r'\d+')
text = "Numbers: 123, 456, 789"
matches = regex.findall(text)
print(matches)  # 输出: ['123', '456', '789']
```


```python
## 从markdown中提取python代码 
import re 
def get_pythoncode_from_markdown(text): 
	pattern = r'python\s+(.*?)\s+```' 
	match = re.search(pattern, text, flags=re.DOTALL) 
	if match: 
		code = match.group(1) 
		print("提取的代码为：") 
		print(code) 
	else: print("未能提取到代码。") 
		exit(1) 
		return code 

code = get_pythoncode_from_markdown(text)
```


"""