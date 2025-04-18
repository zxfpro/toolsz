""" 一个方便的变量调试工具 """
import os
import dill

def pull():
    """从临时区加载到当前全局变量"""
    with open(os.getenv("VARIABLE_PATH"), "rb") as f:
        loaded_vars = dill.load(f)
    # 将变量注入到当前命名空间
    globals().update(loaded_vars)

def push():
    """将当前变量推送到临时区"""
    with open(os.getenv("VARIABLE_PATH"), "wb") as f:
        dill.dump({k:v for k,v in globals().items() if not k.startswith('__')}, f)
