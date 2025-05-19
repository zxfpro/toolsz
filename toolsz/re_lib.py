""" 一些常用的正则工具"""
import re
from abc import ABC, abstractmethod

class RegexExtractor(ABC):
    """正则提取器抽象基类"""

    @abstractmethod
    def extract(self, text: str):
        pass

class PythonCodeExtractor(RegexExtractor):
    """提取markdown中的python代码"""

    def extract(self, text: str):
        pattern = r'```python([\s\S]*?)```'
        matches = re.findall(pattern, text)
        return matches

class JsonCodeExtractor(RegexExtractor):
    """提取markdown中的json代码"""

    def extract(self, text: str):
        pattern = r'```json([\s\S]*?)```'
        matches = re.findall(pattern, text)
        return matches
    

"""

import re
code_match = re.search(r'### 3. Function\n\n```python(.*?)```',python_code,flags = re.DOTALL)

# %% 切分 %%
text = "Hello, World! How are you?"
split_result = re.split(r'\W+', text)
print(split_result)  # 输出: ['Hello', 'World', 'How', 'are', 'you']

# 内容替换
# 使用 re.sub() 替换内容
text = "Hello 123 World 456"
cleaned_text = re.sub(r'\d+', '', text)  # 替换所有数字为空字符串
print(cleaned_text)  # 输出: "Hello  World  "

regex = re.compile(r'\d+')


"""

class RegexExtractorFactory:
    """正则提取器工厂"""

    _extractors = {
        "python": PythonCodeExtractor,
        "json": JsonCodeExtractor,
    }

    @staticmethod
    def get_extractor(code_type: str) -> RegexExtractor:
        extractor_cls = RegexExtractorFactory._extractors.get(code_type.lower())
        if extractor_cls is None:
            raise ValueError(f"不支持的提取类型: {code_type}")
        return extractor_cls()

# 用法示例
# extractor = RegexExtractorFactory.get_extractor("python")
# python_codes = extractor.extract(text)

# extractor = RegexExtractorFactory.get_extractor("json")
# json_codes = extractor.extract(text)



