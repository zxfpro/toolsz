"""
项目管理工具架构设计

该工具旨在支持用户与大模型进行多轮对话，并对项目进行版本管理和维护。
用户可以在特定版本的基础上进行变更和更改。

架构概览：

- DialogueManager: 管理用户与模型之间的对话历史。
- VersionManager: 管理项目的不同版本。
- ChangeApplier: 在选定的版本基础上应用用户的变更指令。
- CLI: 提供命令行交互界面。
- Project: 表示一个项目，包含当前状态和版本历史。

设计模式：

- 仓库模式: VersionManager 作为版本仓库。
- 命令模式: 用户的操作可以被封装为命令。
- 观察者模式: 版本变更时通知相关组件。
"""
import argparse
import os
import json
import uuid
from datetime import datetime

class DialogueManager:
    """管理用户与模型之间的对话历史"""
    def __init__(self, project_id):
        self.project_id = project_id
        self.history = [] # 存储对话消息

    def add_message(self, role, content):
        """添加对话消息"""
        self.history.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})

    def get_history(self):
        """获取对话历史"""
        return self.history

    def save_history(self, file_path):
        """保存对话历史到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=4)

    def load_history(self, file_path):
        """从文件加载对话历史"""
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                self.history = json.load(f)

class VersionManager:
    """管理项目的不同版本"""
    def __init__(self, project_path):
        self.project_path = project_path
        self.versions_dir = os.path.join(project_path, ".versions")
        os.makedirs(self.versions_dir, exist_ok=True)
        self.current_version = None # 当前工作版本

    def create_version(self, version_name, project_state):
        """创建新版本"""
        version_id = str(uuid.uuid4())
        version_path = os.path.join(self.versions_dir, version_id)
        os.makedirs(version_path, exist_ok=True)

        # 保存项目状态 (这里简化为保存一个状态文件)
        state_file = os.path.join(version_path, "state.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(project_state, f, ensure_ascii=False, indent=4)

        version_info = {
            "id": version_id,
            "name": version_name,
            "timestamp": datetime.now().isoformat(),
            "path": version_path
        }
        self._save_version_info(version_id, version_info)
        self.current_version = version_info
        print(f"版本 '{version_name}' ({version_id}) 创建成功。")
        return version_info

    def _save_version_info(self, version_id, version_info):
        """保存版本信息"""
        info_file = os.path.join(self.versions_dir, f"{version_id}.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, ensure_ascii=False, indent=4)

    def _load_version_info(self, version_id):
        """加载版本信息"""
        info_file = os.path.join(self.versions_dir, f"{version_id}.json")
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def list_versions(self):
        """列出所有版本"""
        versions = []
        for filename in os.listdir(self.versions_dir):
            if filename.endswith(".json"):
                version_id = filename[:-5]
                version_info = self._load_version_info(version_id)
                if version_info:
                    versions.append(version_info)
        return sorted(versions, key=lambda x: x.get("timestamp"))

    def checkout_version(self, version_identifier):
        """切换到指定版本"""
        versions = self.list_versions()
        target_version = None
        for version in versions:
            if version["id"] == version_identifier or version["name"] == version_identifier:
                target_version = version
                break

        if target_version:
            self.current_version = target_version
            print(f"已切换到版本 '{target_version['name']}' ({target_version['id']})。")
            # TODO: 实际加载版本状态到工作区
            return target_version
        else:
            print(f"未找到版本 '{version_identifier}'。")
            return None

    def get_current_version(self):
        """获取当前工作版本"""
        return self.current_version

class ChangeApplier:
    """在选定的版本基础上应用用户的变更指令"""
    def __init__(self, project):
        self.project = project

    def apply_changes(self, changes_instruction):
        """应用变更指令"""
        current_state = self.project.get_current_state()
        print(f"在当前版本基础上应用变更：{changes_instruction}")

        # 简化处理：解析形如 "set key=value" 的指令
        if changes_instruction.startswith("set "):
            try:
                key_value = changes_instruction[4:].split("=", 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    current_state[key] = value
                    print(f"已设置状态：{key} = {value}")
                else:
                    print("无效的 set 指令格式。请使用 'set key=value'。")
            except Exception as e:
                print(f"应用变更时发生错误: {e}")
        else:
            print("不支持的变更指令格式。")

        # TODO: 更复杂的变更逻辑需要与大模型交互

        return current_state

class Project:
    """表示一个项目，包含当前状态和版本历史"""
    def __init__(self, project_id, project_path):
        self.project_id = project_id
        self.project_path = project_path
        self.dialogue_manager = DialogueManager(project_id)
        self.version_manager = VersionManager(project_path)
        self.change_applier = ChangeApplier(self)
        self.current_state = {} # 当前项目状态 (例如，文件内容、配置等)
        self._load_project_state()

    def _save_project_state(self):
        """保存项目状态到文件"""
        state_file = os.path.join(self.project_path, "current_state.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_state, f, ensure_ascii=False, indent=4)
        print("项目状态已保存。")

    def _load_project_state(self):
        """加载项目状态"""
        state_file = os.path.join(self.project_path, "current_state.json")
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                self.current_state = json.load(f)
            print("项目状态已加载。")
        else:
            self.current_state = {}
            print("未找到项目状态文件，初始化为空状态。")

    def get_current_state(self):
        """获取当前项目状态"""
        return self.current_state

    def set_current_state(self, state):
        """设置当前项目状态"""
        self.current_state = state

    def create_version(self, version_name):
        """创建新版本"""
        return self.version_manager.create_version(version_name, self.current_state)

    def checkout_version(self, version_identifier):
        """切换到指定版本"""
        return self.version_manager.checkout_version(version_identifier)

    def list_versions(self):
        """列出所有版本"""
        return self.version_manager.list_versions()

    def dialogue(self, message):
        """与模型进行对话并记录"""
        self.dialogue_manager.add_message("user", message)
        # TODO: 与大模型交互获取回复
        model_response = f"模型回复：{message}" # 模拟模型回复
        self.dialogue_manager.add_message("model", model_response)
        print(f"模型回复：{model_response}")
        return model_response

    def apply_changes(self, changes_instruction):
        """应用变更指令"""
        new_state = self.change_applier.apply_changes(changes_instruction)
        self.set_current_state(new_state)
        self._save_project_state()

class CLI:
    """命令行交互界面"""
    def __init__(self):
        self.current_project = None
        self.projects_dir = "projects" # 存储项目的目录
        os.makedirs(self.projects_dir, exist_ok=True)

    def execute(self, args):
        """根据解析的参数执行相应的命令"""
        if args.command == "create":
            self.create_project(args.project_name)
        elif args.command == "open":
            self.open_project(args.project_identifier)
        elif args.command == "version":
            self.create_version(args.version_name)
        elif args.command == "checkout":
            self.checkout_version(args.version_identifier)
        elif args.command == "list_versions":
            self.list_versions()
        elif args.command == "dialogue":
            self.dialogue(args.message)
        elif args.command == "apply":
            self.apply_changes(args.changes_instruction)
        elif args.command == "list_projects":
            self.list_projects()
        elif args.command == "delete_project":
            self.delete_project(args.project_identifier)
        elif args.command == "delete_version":
            self.delete_version(args.version_identifier)
        # exit 命令不再需要，因为程序执行完一个命令就退出
        else:
            print("未知命令。")

    def create_project(self, project_name):
        """创建新项目"""
        if not project_name:
            print("请提供项目名称。")
            return

        project_id = str(uuid.uuid4())
        project_path = os.path.join(self.projects_dir, project_id)
        os.makedirs(project_path, exist_ok=True)

        # 保存项目信息
        project_info = {"id": project_id, "name": project_name}
        with open(os.path.join(project_path, "project_info.json"), 'w', encoding='utf-8') as f:
            json.dump(project_info, f, ensure_ascii=False, indent=4)

        self.current_project = Project(project_id, project_path)
        print(f"项目 '{project_name}' ({project_id}) 创建成功。")

    def list_projects(self):
        """列出所有项目"""
        projects = []
        for project_id in os.listdir(self.projects_dir):
            project_path = os.path.join(self.projects_dir, project_id)
            if os.path.isdir(project_path):
                info_file = os.path.join(project_path, "project_info.json")
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        project_info = json.load(f)
                        projects.append(project_info)
        if projects:
            print("可用项目：")
            for project in projects:
                print(f"- {project['name']} ({project['id']})")
        else:
            print("没有找到任何项目。")
        return []

    def open_project(self, project_identifier):
        """打开现有项目"""
        projects = self.list_projects()
        target_project_info = None
        for project_info in projects:
            if project_info["id"] == project_identifier or project_info["name"] == project_identifier:
                target_project_info = project_info
                break

        if target_project_info:
            project_path = os.path.join(self.projects_dir, target_project_info["id"])
            self.current_project = Project(target_project_info["id"], project_path)
            print(f"已打开项目 '{target_project_info['name']}' ({target_project_info['id']})。")
            # TODO: 加载项目状态和对话历史
            return self.current_project
        else:
            print(f"未找到项目 '{project_identifier}'。")
            return None

    def create_version(self, version_name):
        """创建新版本"""
        if not self.current_project:
            print("请先创建或打开一个项目。")
            return
        if not version_name:
            print("请提供版本名称。")
            return
        self.current_project.create_version(version_name)

    def checkout_version(self, version_identifier):
        """切换到指定版本"""
        if not self.current_project:
            print("请先创建或打开一个项目。")
            return
        if not version_identifier:
            print("请提供版本名称或ID。")
            return
        self.current_project.checkout_version(version_identifier)

    def list_versions(self):
        """列出所有版本"""
        if not self.current_project:
            print("请先创建或打开一个项目。")
            return
        versions = self.current_project.list_versions()
        if versions:
            print("项目版本：")
            for version in versions:
                print(f"- {version['name']} ({version['id']}) - {version['timestamp']}")
        else:
            print("当前项目没有版本。")

    def dialogue(self, message):
        """与模型进行对话并记录"""
        if not self.current_project:
            print("请先创建或打开一个项目。")
            return
        if not message:
            print("请输入对话内容。")
            return
        self.current_project.dialogue(message)

    def apply_changes(self, changes_instruction):
        """应用变更指令"""
        if not self.current_project:
            print("请先创建或打开一个项目。")
            return
        if not changes_instruction:
            print("请输入变更指令。")
            return
        self.current_project.apply_changes(changes_instruction)

    def delete_project(self, project_identifier):
        """删除项目"""
        if not project_identifier:
            print("请提供要删除的项目名称或ID。")
            return

        projects = self.list_projects()
        target_project_info = None
        for project_info in projects:
            if project_info["id"] == project_identifier or project_info["name"] == project_identifier:
                target_project_info = project_info
                break

        if target_project_info:
            if self.current_project and self.current_project.project_id == target_project_info["id"]:
                print("无法删除当前正在使用的项目，请先切换到其他项目。")
                return

            project_path = os.path.join(self.projects_dir, target_project_info["id"])
            try:
                import shutil
                shutil.rmtree(project_path)
                print(f"项目 '{target_project_info['name']}' ({target_project_info['id']}) 已删除。")
            except OSError as e:
                print(f"删除项目时发生错误: {e}")
        else:
            print(f"未找到项目 '{project_identifier}'。")

    def delete_version(self, version_identifier):
        """删除版本"""
        if not self.current_project:
            print("请先创建或打开一个项目。")
            return
        if not version_identifier:
            print("请提供要删除的版本名称或ID。")
            return

        versions = self.current_project.version_manager.list_versions()
        target_version = None
        for version in versions:
            if version["id"] == version_identifier or version["name"] == version_identifier:
                target_version = version
                break

        if target_version:
            if self.current_project.version_manager.current_version and \
               self.current_project.version_manager.current_version["id"] == target_version["id"]:
                print("无法删除当前工作版本。")
                return

            try:
                import shutil
                shutil.rmtree(target_version["path"])
                # 删除版本信息文件
                info_file = os.path.join(self.current_project.version_manager.versions_dir, f"{target_version['id']}.json")
                if os.path.exists(info_file):
                    os.remove(info_file)
                print(f"版本 '{target_version['name']}' ({target_version['id']}) 已删除。")
            except OSError as e:
                print(f"删除版本时发生错误: {e}")
        else:
            print(f"未找到版本 '{version_identifier}'。")

# 伪代码运行入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="项目管理工具 CLI")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # create 命令
    create_parser = subparsers.add_parser("create", help="创建一个新项目")
    create_parser.add_argument("project_name", help="项目名称")

    # open 命令
    open_parser = subparsers.add_parser("open", help="打开一个现有项目")
    open_parser.add_argument("project_identifier", help="项目名称或ID")

    # version 命令
    version_parser = subparsers.add_parser("version", help="为当前项目创建一个新版本")
    version_parser.add_argument("version_name", help="版本名称")

    # checkout 命令
    checkout_parser = subparsers.add_parser("checkout", help="切换到指定版本")
    checkout_parser.add_argument("version_identifier", help="版本名称或ID")

    # list_versions 命令
    subparsers.add_parser("list_versions", help="列出当前项目的所有版本")

    # dialogue 命令
    dialogue_parser = subparsers.add_parser("dialogue", help="与模型进行对话并记录")
    dialogue_parser.add_argument("message", help="对话内容")

    # apply 命令
    apply_parser = subparsers.add_parser("apply", help="在当前版本基础上应用变更")
    apply_parser.add_argument("changes_instruction", help="变更指令")

    # list_projects 命令
    subparsers.add_parser("list_projects", help="列出所有项目")

    # delete_project 命令
    delete_project_parser = subparsers.add_parser("delete_project", help="删除指定项目")
    delete_project_parser.add_argument("project_identifier", help="项目名称或ID")

    # delete_version 命令
    delete_version_parser = subparsers.add_parser("delete_version", help="删除指定版本")
    delete_version_parser.add_argument("version_identifier", help="版本名称或ID")

    args = parser.parse_args()

    cli = CLI()
    if args.command:
        cli.execute(args)
    else:
        parser.print_help()