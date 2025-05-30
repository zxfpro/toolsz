
```python

from llama_index.tools import BaseTool, ToolMetadata
from typing import Optional, List, Dict, Any
import paramiko
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import json
import os

# ShellTools 相关工具
class InteractionXshellTool(BaseTool):
    """交互式 shell 工具，用于执行命令并获取结果"""
    metadata: ToolMetadata = ToolMetadata(
        name="interaction_xshell",
        description="This is an interactive shell tool,It remembers your actions and allows you to take them step by step."
    )
    
    def __init__(
        self,
        ip: str = '127.0.0.1',
        username: str = 'zxf',
        password: str = 'password',
        supervision: bool = True,
        verbose: bool = False,
        log_path: str = '~/.zxf_file/cache/ai_shell.log',
        print_level: str = "INFO"
    ):
        self.ip = ip
        self.username = username
        self.password = password
        self.supervision = supervision
        self.verbose = verbose
        self.log_path = log_path
        self.print_level = print_level
        self.ssh = None
        self.channel = None
        self._login()
    
    def _login(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(self.ip, port=22, username=self.username, password=self.password)
        trans = self.ssh.get_transport()
        self.channel = trans.open_channel("session")
        self.channel.get_pty()
        self.channel.invoke_shell()
        time.sleep(1)
    
    def _guard(self, command: str) -> None:
        for order in ['rm ', 'mv ', 'ln ']:
            if order in command:
                k = input(f'ask for : {command} y/n')
                if k == 'n':
                    raise "退出"
    
    def _get_result(self, interval=1):
        result_all = ''
        while True:
            time.sleep(interval)
            if self.channel.recv_ready():
                output = self.channel.recv(65535)
                result = output.decode('utf-8')
                result_all += result
                if self.verbose:
                    print(f"\033[95m{result}\033[0m")
            else:
                break
        return result_all
    
    def run(self, command: str) -> str:
        if self.supervision:
            self._guard(command)
        self.channel.send(f'{command}\n')
        if 'install' in command:
            result = self._get_result(2)
        else:
            result = self._get_result(1)
        if len(result) > 1000:
            result_all = result[:500] + '......' + result[-500:]
        else:
            result_all = result
        return result_all

class WaitTool(BaseTool):
    """等待工具，用于让工具等待指定秒数"""
    metadata: ToolMetadata = ToolMetadata(
        name="wait",
        description="this tool function is wait for the tool to finish executing the command. you can wait t seconds"
    )
    
    def run(self, t: str) -> str:
        time.sleep(int(t))
        return f"{t} seconds later"

# CoderTools 相关工具
class CodeInterpreterTool(BaseTool):
    """代码解释器工具，用于执行 Python 代码并返回输出和错误"""
    metadata: ToolMetadata = ToolMetadata(
        name="code_interpreter",
        description="A function to execute python code, and return the stdout and stderr"
    )
    
    def __init__(self):
        from llama_hub.tools.code_interpreter import CodeInterpreterToolSpec
        self.inter = CodeInterpreterToolSpec()
    
    def run(self, code_path: str) -> str:
        with open(code_path, 'r') as f:
            code = f.read()
        return self.inter.code_interpreter(code)

class WriteTestCodeTool(BaseTool):
    """编写测试代码工具，用于为给定代码编写完整的测试代码"""
    metadata: ToolMetadata = ToolMetadata(
        name="write_test_code",
        description="This is a tool for writing complete test code for a given piece of code"
    )
    
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, code: str) -> str:
        result = self.llm.complete(f"""
        Write test code for this code, which should be as complete as possible to cover all parameter combinations
        code:
        {code}
        -----------
        text_code:
        """).text
        return result

class RunTestCodeTool(BaseTool):
    """运行测试代码工具，用于执行测试代码"""
    metadata: ToolMetadata = ToolMetadata(
        name="run_test_code",
        description="This tool helps you execute your test code locally"
    )
    
    def __init__(self):
        from llama_hub.tools.code_interpreter import CodeInterpreterToolSpec
        self.inter = CodeInterpreterToolSpec()
    
    def run(self, code: str) -> str:
        return self.inter.code_interpreter(code)

class RewriteTestCodeTool(BaseTool):
    """重写测试代码工具，根据要求重写测试代码"""
    metadata: ToolMetadata = ToolMetadata(
        name="rewrite_test_code",
        description="This is a tool that allows you to rewrite test code according to guidance and requirements"
    )
    
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, test_code: str, demand: str) -> str:
        result = self.llm.complete(f"""
        Modify the existing test code as required.
        require: {demand}
        test_code:
        {test_code}
        -----------
        new_text_code:
        """).text
        return result

class CodeWriterTool(BaseTool):
    """代码编写工具，根据要求编写 Python 代码"""
    metadata: ToolMetadata = ToolMetadata(
        name="code_writer",
        description="You are a Python code expert and can write code according to requirements."
    )
    
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, text: str) -> str:
        return self.llm.complete(
            f'You are a Python code expert and can write code according to requirements. Requirement: {text}'
        ).text

# FileTools 相关工具
class DelayTool(BaseTool):
    """延迟工具，用于跳过指定秒数"""
    metadata: ToolMetadata = ToolMetadata(
        name="delay",
        description="the tool can be used to jump over [time_later] second"
    )
    
    def run(self, time_later: int) -> str:
        time.sleep(time_later)
        return f'{time_later} second later'

class SavePyTool(BaseTool):
    """保存 Python 代码工具，将代码保存到本地文件"""
    metadata: ToolMetadata = ToolMetadata(
        name="save_py",
        description="This is a code review tool that saves your code to a local file"
    )
    
    def run(self, code: str, save_path: str) -> str:
        with open(save_path, 'w') as f:
            f.write(code)
        return f'saved your code in {save_path}'

class ReadPyTool(BaseTool):
    """读取 Python 文件工具，读取指定路径的 Python 文件内容"""
    metadata: ToolMetadata = ToolMetadata(
        name="read_py",
        description="这个工具可以用来读py文件 输入文件路径 就能得到文件中的内容"
    )
    
    def run(self, path: str) -> str:
        try:
            with open(path, 'r') as f:
                text = f.read()
        except Exception as e:
            return f'Error {e}'
        return text

# HumanIOTools 相关工具
class HumanReviewerTool(BaseTool):
    """人工审核工具，用于在最终提交前验证工作是否符合要求"""
    metadata: ToolMetadata = ToolMetadata(
        name="human_reviewer",
        description="Use it before the final submission to verify if your work meets the requirements."
    )
    
    def run(self, text: str) -> str:
        print('***********check************')
        result = input(text)
        if result == 'q':
            raise Exception('终止任务')
        return result

class DemandConfirmationTool(BaseTool):
    """需求确认工具，用于在需求不明确时向用户提问以澄清需求"""
    metadata: ToolMetadata = ToolMetadata(
        name="demand_confirmation",
        description="If you feel that the requirements are unclear, you can use the tool to ask the user questions to clarify the requirements"
    )
    
    def run(self, question: str) -> str:
        result = input(question)
        return result

class QuestionRewriterTool(BaseTool):
    """问题重写工具，用于根据背景信息和用户问题生成更具体的问题描述"""
    metadata: ToolMetadata = ToolMetadata(
        name="question_rewriter",
        description="If the user's question is not clear enough, you can use this tool to get a clearer question description."
    )
    
    def __init__(self, llm):
        self.llm = llm
    
    def run(self, question: str) -> str:
        result = self.llm.complete(f"""
        Please provide a more specific and complete description of the user's question based on the background:
        -------------------
        and the question asked by the user:
        {question}
        -------------------
        Make the user's problem more specific and complete
        """)
        return result.text

# MathTools 相关工具
class MultiplyTool(BaseTool):
    """乘法工具，计算两个整数的乘积"""
    metadata: ToolMetadata = ToolMetadata(
        name="multiply",
        description="Multiple two integers and returns the result integer"
    )
    
    def run(self, a: int, b: int) -> int:
        return a * b

class MinusTool(BaseTool):
    """减法工具，计算两个整数的差"""
    metadata: ToolMetadata = ToolMetadata(
        name="minus",
        description="minus two integers and returns the result integer"
    )
    
    def run(self, a: int, b: int) -> int:
        return a - b

class PlusTool(BaseTool):
    """加法工具，计算两个整数的和"""
    metadata: ToolMetadata = ToolMetadata(
        name="plus",
        description="plus two integers and returns the result integer"
    )
    
    def run(self, a: int, b: int) -> int:
        return a + b

# WebTools 相关工具
class GetAllLinksTool(BaseTool):
    """获取所有链接工具，从给定 URL 中提取所有链接"""
    metadata: ToolMetadata = ToolMetadata(
        name="get_all_links",
        description="Retrieve all links from a given URL and return a list of tuples containing the link text and the absolute URL"
    )
    
    def run(self, url: str) -> List[tuple]:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the page: {url}")
            return []
        soup = BeautifulSoup(response.content, "html.parser")
        links = [
            (a.text, urljoin(url, a["href"]))
            for a in soup.find_all("a", href=True)
            if a["href"]
        ]
        return links

class GetContentTool(BaseTool):
    """获取内容工具，从给定 URL 中提取文本内容"""
    metadata: ToolMetadata = ToolMetadata(
        name="get_content",
        description="Retrieve the text content from a given URL and return it as a string"
    )
    
    def run(self, url: str) -> str:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the page: {url}")
            return ""
        soup = BeautifulSoup(response.content, "html.parser")
        text_content = soup.get_text()
        return text_content

# ExaTools 相关工具
class SearchTool(BaseTool):
    """搜索工具，根据查询搜索网页"""
    metadata: ToolMetadata = ToolMetadata(
        name="search",
        description="Search for a webpage based on the query."
    )
    
    def __init__(self):
        from exa_py import Exa
        self.exa = Exa(api_key=os.environ["EXA_API_KEY"])
    
    def run(self, query: str) -> List[Dict[str, Any]]:
        return self.exa.search(f"{query}", use_autoprompt=True, num_results=5)

class FindSimilarTool(BaseTool):
    """查找相似工具，查找与给定 URL 类似的网页"""
    metadata: ToolMetadata = ToolMetadata(
        name="find_similar",
        description="Search for webpages similar to a given URL."
    )
    
    def __init__(self):
        from exa_py import Exa
        self.exa = Exa(api_key=os.environ["EXA_API_KEY"])
    
    def run(self, url: str) -> List[Dict[str, Any]]:
        return self.exa.find_similar(url, num_results=5)

class GetContentsTool(BaseTool):
    """获取内容工具，获取网页内容"""
    metadata: ToolMetadata = ToolMetadata(
        name="get_contents",
        description="Get the contents of a webpage."
    )
    
    def __init__(self):
        from exa_py import Exa
        self.exa = Exa(api_key=os.environ["EXA_API_KEY"])
    
    def run(self, ids: list[str]) -> List[Dict[str, Any]]:
        return self.exa.get_contents(ids)

# IssueAssignments 相关工具
class IssueSweepingAssignmentsTool(BaseTool):
    """发布扫地任务工具"""
    metadata: ToolMetadata = ToolMetadata(
        name="issue_sweeping_assignments",
        description="您可以通过此工具来发布一个扫地任务"
    )
    
    def run(self, text: str) -> str:
        data = {"text": text, "token": "57d3d17868206f5e181fd27d0cbdde89d9739b0538b27ddd"}
        json_data = json.dumps(data)
        r = requests.post('http://127.0.0.1:9000/sweep', data=json_data)
        return json.loads(r.text).get('msg')

class IssueCodingAssignmentsTool(BaseTool):
    """发布代码生成任务工具"""
    metadata: ToolMetadata = ToolMetadata(
        name="issue_coding_assignments",
        description="您可以通过此工具来发布一个代码生成任务"
    )
    
    def run(self, text: str) -> str:
        data = {"text": text, "token": "57d3d17868206f5e181fd27d0cbdde89d9739b0538b27ddd"}
        json_data = json.dumps(data)
        r = requests.post('http://127.0.0.1:9000/code', data=json_data)
        return json.loads(r.text).get('msg')

class IssueEnvironmentBuildingAssignmentsTool(BaseTool):
    """发布环境构建任务工具"""
    metadata: ToolMetadata = ToolMetadata(
        name="issue_environment_building_assignments",
        description="您可以通过此工具来发布一个环境构建任务"
    )
    
    def run(self, text: str) -> str:
        data = {"text": text, "token": "57d3d17868206f5e181fd27d0cbdde89d9739b0538b27ddd"}
        json_data = json.dumps(data)
        r = requests.post('http://127.0.0.1:9000/environment', data=json_data)
        return json.loads(r.text).get('msg')

class IssueInformationCollectionAssignmentsTool(BaseTool):
    """发布信息收集任务工具"""
    metadata: ToolMetadata = ToolMetadata(
        name="issue_information_collection_assignments",
        description="您可以通过此工具来发布一个信息收集任务"
    )
    
    def run(self, topic: str) -> str:
        data = {"text": topic, "token": "57d3d17868206f5e181fd27d0cbdde89d9739b0538b27ddd"}
        json_data = json.dumps(data)
        r = requests.post('http://127.0.0.1:9000/information', data=json_data)
        return json.loads(r.text).get('msg')

class IssueWritingAssignmentsTool(BaseTool):
    """发布写作任务工具"""
    metadata: ToolMetadata = ToolMetadata(
        name="issue_writing_assignments",
        description="您可以通过此工具来发布一个写作任务"
    )
    
    def run(self, topic: str, infos: str) -> str:
        data = {"text": topic, "token": "57d3d17868206f5e181fd27d0cbdde89d9739b0538b27ddd"}
        json_data = json.dumps(data)
        r = requests.post('http://127.0.0.1:9000/information', data=json_data)
        return json.loads(r.text).get('msg')
```

```python

import subprocess

def execute_python_code(code):
    """
    执行传入的 Python 代码，返回输出和错误信息。
    """
    try:
        # 使用 subprocess 执行 Python 脚本
        process = subprocess.run(["python3", "-c", code], capture_output=True, text=True)
        return process.stdout, process.stderr
    except Exception as e:
        return "", str(e)

def execute_shell_command(command):
    """
    执行传入的 Shell 命令，返回输出和错误信息。
    """
    try:
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        return process.stdout, process.stderr
    except Exception as e:
        return "", str(e)
```

```python

from llama_index.core.tools import BaseTool,ToolMetadata

from typing import Optional, List, Dict, Any
import paramiko
import time
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import json
import os

class InteractionXshellTool():
    """交互式 shell 工具，用于执行命令并获取结果"""
    metadata: ToolMetadata = ToolMetadata(
        name="interaction_xshell",
        description="This is an interactive shell tool,It remembers your actions and allows you to take them step by step."
    )
    
    def __init__(
        self,
        ip: str = '127.0.0.1',
        username: str = 'zxf',
        password: str = 'password',
        supervision: bool = True,
        verbose: bool = False,
        log_path: str = '~/.zxf_file/cache/ai_shell.log',
        print_level: str = "INFO"
    ):
        self.ip = ip
        self.username = username
        self.password = password
        self.supervision = supervision
        self.verbose = verbose
        self.log_path = log_path
        self.print_level = print_level
        self.ssh = None
        self.channel = None
        self._login()
    
    def _login(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(self.ip, port=22, username=self.username, password=self.password)
        trans = self.ssh.get_transport()
        self.channel = trans.open_channel("session")
        self.channel.get_pty()
        self.channel.invoke_shell()
        time.sleep(1)
    
    def _guard(self, command: str) -> None:
        for order in ['rm ', 'mv ', 'ln ']:
            if order in command:
                k = input(f'ask for : {command} y/n')
                if k == 'n':
                    raise "退出"
    
    def _get_result(self, interval=1):
        result_all = ''
        while True:
            time.sleep(interval)
            if self.channel.recv_ready():
                output = self.channel.recv(65535)
                result = output.decode('utf-8')
                result_all += result
                if self.verbose:
                    print(f"\033[95m{result}\033[0m")
            else:
                break
        return result_all
    
    def run(self, command: str) -> str:
        if self.supervision:
            self._guard(command)
        self.channel.send(f'{command}\n')
        if 'install' in command:
            result = self._get_result(2)
        else:
            result = self._get_result(1)
        if len(result) > 1000:
            result_all = result[:500] + '......' + result[-500:]
        else:
            result_all = result
        return result_all

inter = InteractionXshellTool(ip = '127.0.0.1',
        username  = 'zxf',
        password  = 'password',
        supervision  = True,
        verbose = True,
        log_path = '~/.zxf_file/cache/ai_shell.log',
        print_level = "INFO")
        
```



```python

class WebTools(BaseTools):
    register_default = ['get_all_links', 'get_content']
    def __init__(self, register=[]):
        super().__init__(register, self.register_default)
    
    def get_all_links(self, url):
        """Retrieve all links from a given URL and return a list of tuples containing the link text and the absolute URL"""
        response = requests.get(url)
        if response.status_code != 200:
        print(f"Failed to retrieve the page: {url}")
        return []

        soup = BeautifulSoup(response.content, "html.parser")
        links = [
        (a.text, urljoin(url, a["href"]))
        for a in soup.find_all("a", href=True)
        if a["href"]
        ]
        return links

    def get_content(self, url):
        """Retrieve the text content from a given URL and return it as a string"""
        response = requests.get(url)
        if response.status_code != 200:
        print(f"Failed to retrieve the page: {url}")
        return []
        soup = BeautifulSoup(response.content, "html.parser")
        text_content = soup.get_text()
        return text_content


class ExaTools(BaseTools):
    register_default = ['search', 'find_similar', 'get_contents']

    def __init__(self, register=[]):
        super().__init__(register, self.register_default)
        from exa_py import Exa
        self.exa = Exa(api_key=os.environ["EXA_API_KEY"])

    def search(self, query: str):
        """Search for a webpage based on the query."""
        return self.exa.search(f"{query}", use_autoprompt=True, num_results=5)

    def find_similar(self, url: str):
        """Search for webpages similar to a given URL.
        The url passed in should be a URL returned from `search`.
        """
        return self.exa.find_similar(url, num_results=5)

    def get_contents(self, ids: list[str]):
        """Get the contents of a webpage.
        The ids passed in should be a list of ids returned from `search`.
        """
        return self.exa.get_contents(ids)
```


```python

class FileTools(BaseTools):
    register_default = ['delay', 'save_py', 'read_py']
    def __init__(self, llm, register=[]):
        super().__init__(register,self.register_default)
        self.llm = llm

    def delay(self,time_later: int) -> str:
        """
        the tool can be used to jump over [time_later] second
        """
        import time
        time.sleep(time_later)
        return f'{time_later} second later'



    def save_py(self,code: str, save_path: str) -> str:

        """
        This is a code review tool that saves your code to a local file so that you can use shell_tool to validate your code in a local dockerfiles.
        for example:
        save_path = temp.py
        """
        with open(save_path, 'w') as f:
            f.write(code)
        return f'saved your code in {save_path}'



    def read_py(self,path: str) -> str:
        """
        这个工具可以用来读py文件 输入文件路径 就能得到文件中的内容
        """
        try:
            with open(path, 'r') as f:
                text = f.read()
        except Exception as e:
            return f'Error {e}'
        return text
```

```python
from urllib.parse import urljoin
import requests
import json
import os
from bs4 import BeautifulSoup
import paramiko
import time

class CoderTools(BaseTools):
    register_default = ['code_interpreter', 'write_test_code', 'run_test_code', 'rewrite_test_code', 'code_writer']
    def __init__(self, register=[],llm=None):
        super().__init__(register,self.register_default)
        self.llm = llm
        from llama_hub.tools.code_interpreter import CodeInterpreterToolSpec
        self.inter = CodeInterpreterToolSpec()

    def code_interpreter(self,code_path:str)->str:
        """
        A function to execute python code, and return the stdout and stderr
        You should import any libraries that you wish to use. You have access to any libraries the user has installed.
        The code passed to this functuon is executed in isolation. It should be complete at the time it is passed to this function.
        You should interpret the output and errors returned from this function, and attempt to fix any problems.
        If you cannot fix the error, show the code to the user and ask for help
        It is not possible to return graphics or other complicated data from this function. If the user cannot see the output, save it to a file and tell the user.
        """
        with open(code_path,'r') as f:
            code = f.read()
        return self.inter.code_interpreter(code)

    def write_test_code(self,code: str) -> str:
        """
        This is a tool for writing complete test code for a given piece of code
        """
        result = self.llm.complete(f"""
        Write test code for this code, which should be as complete as possible to cover all parameter combinations
        code:
        {code}
        -----------
        text_code:
        """).text
        return result

    def run_test_code(self,code: str) -> str:
        """
        This tool helps you execute your test code locally
        """
        return self.inter.code_interpreter(code)

    def rewrite_test_code(self,test_code:str,demand:str)->str:
        """
        This is a tool that allows you to rewrite test code according to guidance and requirements
        """
        result = self.llm.complete(f"""
        Modify the existing test code as required.
        require: {demand}
        test_code:
        {test_code}
        -----------
        new_text_code:
        """).text
        return result
    
    def code_writer(self, text: str) -> str:
        return self.llm.complete(
        f'You are a Python code expert and can write code according to requirements. Requirement: {text}').text
```

```python

class ShellTools(BaseTools):

	register_default = ["interaction_xshell", "wait"]

  

def __init__(self, ip='127.0.0.1', username='zxf', password='password', supervision=True, register=[],

    verbose=False, log_path='~/.zxf_file/cache/ai_shell.log',print_level= "INFO"):

    # Sudo apt-get install openssh-server

    # Sudo service ssh start

    super().__init__(register, self.register_default)

    self.ssh = None

    self.channel = None

    self.supervision = supervision

    self.verbose = verbose

    self.logger = Logger(log_path, print_level=print_level)

    self._login(ip=ip, username=username, password=password)


  

def _login(self, ip, username, password) -> str:

    """

    login Xshell

    """

    self.ssh = paramiko.SSHClient()

    # 自动接受不在known_hosts文件的主机密钥

    self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # 连接服务器

    self.ssh.connect(ip, port=22, username=username, password=password)

    # 创建一个交互式的shell会话

    trans = self.ssh.get_transport()

    self.channel = trans.open_channel("session")

    self.channel.get_pty()

    self.channel.invoke_shell()



    # 等待通道准备好

    time.sleep(1)

    self.logger.info('log success')

    return 0

  

def _guard(self, command: str) -> None:

    for order in ['rm ', 'mv ', 'ln ']:

    if order in command:

    k = input(f'ask for : {command} y/n')

    if k == 'n':

    raise "退出"

    k = input(f'ask for commend : {command} y/n')

    if k == 'n':

        raise "退出"

  

def _get_result(self, interval=1):

    result_all = ''

    while True:

        time.sleep(interval)

    if self.channel.recv_ready():

        output = self.channel.recv(65535) # 读取数据

        result = output.decode('utf-8')

        result_all += result

    if self.verbose:

        self.logger.info(f"\033[95m{result}\033[0m")

    else:

        break

    return result_all

  

def interaction_xshell(self, command: str) -> str:

    """

    This is an interactive shell tool,It remembers your actions and allows you to take them step by step.


    if meeting Proceed ([y]/n)? ,you should response.

    """

    if self.supervision:

        self._guard(command)

        # 发送命令

    self.channel.send(f'{command}\n')

    if 'install' in command:

        result = self._get_result(2)

    else:

        result = self._get_result(1)



    if len(result) > 1000:

        result_all = result[:500] + '......' + result[-500:]

    else:

        result_all = result

    return result_all

  

def wait(self, t: str) -> str:
    """
    this tool function is wait for the tool to finish executing the command. you can wait t seconds
    """
    time.sleep(int(t))
    return f"{t} seconds later"

# sl = ShellTools(supervision=False,verbose=True)
# aa = sl.interaction_xshell('conda install -c pytorch -c conda-forge sdv')
```




```python

class CustObsidianReader(BaseReader):
    def load_data(self, file_path: str,
                        extra_info: Optional[Dict] = None) -> List[Document]:
        # 自定义读取逻辑
        with open(file_path, 'r') as file:
            text = file.read()
            
        topic,describe,creation_date,tags,content = get_infos_from_md(text)
            
        # TODO 这里是可以编辑做策略的

        content_cut = content[:4000]
        if len(content_cut) != len(content):
            print(topic,'is too long ***')
        document = Document(text=f"{topic}, describe: {describe}", 
                            metadata={"content":content_cut,
                                      "title":topic,
                                      "tags":tags,},
                            excluded_embed_metadata_keys=["content","tags"])
        return [document]


```