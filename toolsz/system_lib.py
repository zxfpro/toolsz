""" 系统工作 """
import getpass
import json
import os
import requests
import setproctitle

def setprocesstitle(name:str)->None:
    """
    修改线程名

    参数:   
    name (str): 设定的线程名    

    返回:
    None
    """
    setproctitle.setproctitle(name)

def password(key:str):
    """
    动态输入密码

    参数:   
    key (str): 提示信息    

    返回:
    None
    """
    return getpass.getpass(f'{key} input:')


def exec_str(code:str,local_vars:dict = None)->dict:
    """
    执行代码字符串

    参数:   
    code (str): 代码字符串    
    local_vars (dict): 变量   

    返回:
    dict
    """
    if local_vars is None:
        local_vars = {}
    exec(code, globals(), local_vars)
    return local_vars



class DDMessage:
    """DingDing_POST类用于向钉钉发送POST请求。
    """
    def __init__(self):
        """初始化DingDing_POST类。
        """
        token = os.environ.get("DD_TOKEN")
        self.host = f"https://oapi.dingtalk.com/robot/send?access_token={token}"

    def send(self,role: str, content: str)->None:
        """
        向钉钉发送文本消息.   

        参数:   
        role (str): 信号发出者 可以使用Agent System Majordomo    
        content (str): 消息内容。   

        返回:
        None
        """
        assert role in ["Agent", "System", "Majordomo"]
        content = f"{role} : {content}"
        data = {"msgtype": "text", "text": {"content": content}}
        requests.post(
            self.host,
            data=json.dumps(data),
            headers={'Content-Type': 'application/json'}, timeout=10)


def send_message_via_dd(role: str, content: str):
    """
    通过钉钉机器人发送消息。

    参数:   
    role (str): 信号发出者，可以是 "Agent", "System", 或 "Majordomo"。    
    content (str): 要发送的消息内容。

    返回:
    None
    """
    # 确保环境变量中有钉钉机器人的token
    if not os.environ.get("DD_TOKEN"):
        raise ValueError("Environment variable 'DD_TOKEN' is not set.")

    # 创建DDMessage实例
    dd_message = DDMessage()

    # 发送消息
    dd_message.send(role, content)
