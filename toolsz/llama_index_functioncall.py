from llama_index.core.tools.types import ToolOutput


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def xshell(shell:str) -> str:
    "This is an interactive shell tool,It remembers your actions and allows you to take them step by step."
    print(shell,'shell')
    result = input('输入shell:')
    tool_output = ToolOutput(
        content=result,
        tool_name="interaction_xshell",
        raw_input={"args": {"shell":shell}},
        raw_output={"结果": "成功"}
    )
    return tool_output


# 创建工具
from llama_index.core.tools import FunctionTool

def package(fn):
    return FunctionTool.from_defaults(fn=fn)

def package_update(fn,name = None,description = None):

    return FunctionTool.from_defaults(fn=fn,
                                      name = name,
                                      description = description)

# TODO 增加一些MCP能力
