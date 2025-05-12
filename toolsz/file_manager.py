""" 文件工具 """

import os
import ast

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





import re,os,csv,time
import shutil
import pandas as pd
import chardet
from kafka import KafkaProducer
from ..basis.toolbasis import BasisTools


def running_time(func):
    """
    装饰器；
    用于显示程序执行的时间
    """
    def warpper(*args, **mar):
        start = time.time()
        result = func(*args, **mar)
        end = time.time()
        print(f"执行时间是：{end - start}")
        return result
    return warpper

def read_csv(csv_file,type='iter'):
    """
    读取csv文件,以不同的方式读取csv
    :param csv_file:
    :param type:
    :return:
    """
    assert type in ['iter','dict','DataFrame']
    if type =='iter':
        return csv.reader(open(csv_file))
    elif type=='dict':
        with open(csv_file,'r') as f:
            lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        return dict(((name, label) for name, label in tokens))
    elif type =='DataFrame':
        return pd.read_csv(csv_file)

def copyfile(filename, target_dir):
    """将文件复制到目标目录。"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

class FileTools(BasisTools):
    def __init__(self):
        """"""
    def get_encoding_way(self,scv_path):
        """
        识别csv文件的编码格式
        :param scv_path:
        :return:
        """
        f = open(scv_path,'rb')
        data = f.read()
        return chardet.detect(data)
    def get_files(self, path):
        """
        输入一个文件夹路径 输出这个文件夹中的文件 忽略文件夹中的文件夹
        :param path:
        :return:flie_list
        """
        return [os.path.join(path, f) for f in sorted(list(os.listdir(path)))
                if os.path.isfile(os.path.join(path, f))]
    def move_file(self,path1,path2):
        try:
            shutil.move(path1,path2)
        except Exception as e:
            print(e)
    def path_exist(self,file_path):
        """
        判断路径是否存在,不存在则创建,不能跨级创建
        :param file_path:
        :return:
        """
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        return file_path
    def get_two_filepath(self,path,tips='csv'):
        """
        输入一个路径，输出两个路径
        :param path:
        :return:
        """
        cl1p = re.sub(f'.{tips}', f'_cl1.{tips}', path)
        cl2p = re.sub(f'.{tips}', f'_cl2.{tips}', path)
        return cl1p, cl2p


    def safe_save(self,path):
        """
        如果原文件存在,创建一个新的文件名,防止把旧文件覆盖掉.
        :param path:
        :return:
        """
        if os.path.exists(path):
            path = self.get_new_path(path)
        return path
    def get_new_path(self,path):
        """
        获得一个新的文件名的路径
        :param path:
        :return:
        """
        the_parent_directory = '/'.join(path.split('/')[:-1])
        file_path = path.split('/')[-1]
        file_list = os.listdir(the_parent_directory)
        file_list.sort()
        k = file_list[-1].split('.')[0]
        number = k.split('_')[-1]
        try:
            number = int(number) + 1
        except Exception:
            number = 1
        final_name = file_path.split('.')[-1]
        file_name = file_path.split('.')[0].split('_')[0]
        k = f"{file_name}_{number}.{final_name}"
        k = os.path.join(the_parent_directory,k)
        return k


    def make_some_noise(self):
        """
        结束提醒
        :return:
        """
        os.system('spd-say "Now , test finished!"')





    def KafKaProduce(self,bootstrap_servers='192.168.1.27:9092'):
        """
        kafka的使用
        :param bootstrap_servers:卡夫卡地址
        :return:
        """
        producer = KafkaProducer(bootstrap_servers=bootstrap_servers)  # 连接kafka

        msg = "Hello World".encode('utf-8')  # 发送内容,必须是bytes类型
        producer.send('test', msg)  # 发送的topic为test
        producer.close()

    def write_dict(self,number, value_count):
        """
        将一个字典扩增到一定长度，扩增的部分用0补全
        :param number:9
        :param value_count: {"12":12,"34":32}
        :return: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, '12': 12, '34': 32}
        """
        number = number + 1
        dicts = {}
        for item in range(number):
            dicts[item] = 0
        for items, values in value_count.items():
            dicts[items] = values
        return dicts

    def get_split_list(self,number,list):
        """
        将列表分为固定长度的小列表
        :param number: 长度
        :param list: 原列表
        :return:
        """
        columns_list = []
        group = len(list) // number
        for item in range(group):
            columns_list.append(list[number * item:number * (item + 1)])
        columns_list.append(list[number * group:(number * group) + (len(list) % number)])
        return columns_list




























