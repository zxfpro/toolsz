""" 文件工具 """

import os
import ast
import functools
import qrcode as qrcode_
import pickle


class LocalFileTool:
    """
    LocalFileTool TODO
    """
    def __init__(self, base_dir):
        self.base_dir = base_dir
    
    def get_file_list(self, dir_path='', include_extensions=None, exclude_pycache=True):
        """获取指定目录下的所有文件列表
        :param dir_path: 相对于基础目录的子目录路径
        :param include_extensions: 要包含的文件扩展名列表，如 ['.py', '.txt']，默认包含所有文件
        :param exclude_pycache: 是否排除 __pycache__ 目录，默认为 True
        :return: 文件路径列表
        """
        full_dir_path = os.path.join(self.base_dir, dir_path)
        if not os.path.exists(full_dir_path) or not os.path.isdir(full_dir_path):
            print(f"Directory {full_dir_path} does not exist")
            return []
        
        file_list = []
        for root, dirs, files in os.walk(full_dir_path):
            # 排除 __pycache__ 目录
            if exclude_pycache:
                dirs[:] = [d for d in dirs if d != '__pycache__']
            
            relative_root = os.path.relpath(root, self.base_dir)
            for file in files:
                if include_extensions is None or os.path.splitext(file)[1] in include_extensions:
                    relative_path = os.path.join(relative_root, file)
                    # 如果是基础目录本身，避免路径以 ./ 开头
                    if relative_path.startswith(os.path.sep):
                        relative_path = relative_path[len(os.path.sep):]
                    file_list.append(relative_path)
        return file_list
    
    def get_file_content(self, file_path):
        """获取单个文件内容"""
        full_path = os.path.join(self.base_dir, file_path)
        if not os.path.exists(full_path):
            print(f"File {full_path} does not exist")
            return None
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading file {full_path}: {e}")
            return None
    
    def get_multiple_files(self, file_paths):
        """获取多个文件内容并格式化为注释标识的字符串"""
        content_list = []
        for path in file_paths:
            content = self.get_file_content(path)
            if content is not None:
                # 为每个文件添加注释标识
                content_list.append(f"# {path}")
                content_list.append(content)
                # 在文件之间添加分隔符
                content_list.append("\n")
        return "\n".join(content_list)
    
    def write_single_file(self, file_path, content):
        """写入单个文件"""
        full_path = os.path.join(self.base_dir, file_path)
        dir_name = os.path.dirname(full_path)
        
        # 确保目标目录存在
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        
        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing file {full_path}: {e}")
            return False
    
    def write_multiple_files_from_string(self, formatted_string):
        """从格式化字符串写入多个文件"""
        lines = formatted_string.split('\n')
        current_file_path = None
        current_file_content = []
        
        for line in lines:
            if line.startswith('# ') and line.endswith('.py'):
                # 如果之前有文件内容，先写入
                if current_file_path is not None and current_file_content:
                    self.write_single_file(current_file_path, '\n'.join(current_file_content))
                    current_file_content = []
                
                # 提取文件路径
                current_file_path = line[2:].strip()
                # 检查文件路径是否有效
                if not current_file_path:
                    current_file_path = None
                    continue
            elif current_file_path is not None:
                # 添加到当前文件内容
                current_file_content.append(line)
        
        # 写入最后一个文件
        if current_file_path is not None and current_file_content:
            self.write_single_file(current_file_path, '\n'.join(current_file_content))
        
        return True

    def get_related_files(self, file_path):
        """获取与指定文件相关联的所有项目内文件"""
        all_files = self.get_file_list(include_extensions=['.py'])
        related_files = set()
        self._analyze_imports(file_path, all_files, related_files)
        return list(related_files)
    
    def _analyze_imports(self, file_path, all_files, related_files):
        """分析文件的导入语句，递归查找相关文件"""
        if file_path in related_files:
            return
        related_files.add(file_path)
        
        content = self.get_file_content(file_path)
        if content is None:
            return
        
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        self._find_and_add_module(module_name, all_files, related_files)
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module
                    self._find_and_add_module(module_name, all_files, related_files)
        except Exception as e:
            print(f"Error analyzing imports in {file_path}: {e}")
    
    def _find_and_add_module(self, module_name, all_files, related_files):
        """根据模块名查找对应的文件并添加到相关文件列表"""
        for file in all_files:
            if file.endswith(f"{module_name}.py") or file.replace(os.path.sep, '.')[:-3] == module_name:
                self._analyze_imports(file, all_files, related_files)
                break


class PythonLib:
    def __init__(self):
        pass


    @staticmethod
    def input_multiline():
        import sys
        print("请输入内容（按 Ctrl+D 或 Ctrl+Z 后按回车结束输入）：")
        multi_line_input = sys.stdin.read()
        print("你输入的内容是：")
        print(multi_line_input)
        return multi_line_input

    @staticmethod
    def tool_qrcode(img_path: str = './small2.png'):
        """
        用来将函数输出转化为二维码图片的装饰器
        """

        def outer_packing(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                qr = qrcode_.QRCode(version=1, box_size=10, border=5)
                result = func(*args, **kwargs)
                qr.add_data(result)
                qr.make(fit=True)
                img = qr.make_image(fill_color='black', back_color='white')
                img.save(img_path)
                return img
            return wrapper
        return outer_packing
    

""" 函数工具 """
from llama_index.core.tools.types import ToolOutput
from llama_index.core.tools import FunctionTool

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


def package(fn,name:str = None,description:str = None):
    """将一般的函数打包成工具

    Args:
        fn (function): 编写的函数
        name (str, optional): 函数名.
        description (str, optional): 函数描述. Defaults to None.

    Returns:
        FunctionTool: functioncall
    """

    if name is not None or description is not None:
        return FunctionTool.from_defaults(fn=fn,
                                      name = name,
                                      description = description)
    else:
        return FunctionTool.from_defaults(fn=fn)

# TODO 增加一些MCP能力

