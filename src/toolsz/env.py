
class UVManager():
	"""
	uv 是一个现代化的 Python 包管理器和构建工具，旨在提供比 pip 和 venv 更快的性能。

	## 安装
	```bash
	curl -LsSf https://astral.sh/uv/install.sh | sh
	```

	## 命令

	### 初始化项目
	```bash
	uv init .
	uv init <project dir>
	```

	### 添加依赖
	```bash
	# 安装到默认环境
	uv add <package>

	# 安装到开发环境
	uv add --group dev <package>

	# 安装到生产环境
	uv add --group production <package>
	```

	### 移除依赖
	```bash
	uv remove <package>
	```

	### 同步环境
	```bash
	uv sync
	```

	### 构建项目
	```bash
	uv build
	```

	### 运行脚本
	```bash
	uv run ./hallo.py
	```

	### 导出环境
	```bash
	uv export --format requirements-txt > requirements.txt
	```

	### 工具
	```bash
	uv python dir
	uv tool dir
	```

	### 安装python
	```bash
	uv python list
	uv python install 3.13
	```

	### 升级包
	```bash
	uv lock --upgrade-package requests
	```

	### 设定环境变量
	```bash
	# 缓存目录
	export UV_CACHE_DIR=/path/to/cache/dir

	# 镜像地址
	export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

	# 额外镜像地址
	export EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

	# 不用缓存
	export UV_NO_CACHE=0

	# 下载包时的超时时间，单位为秒
	UV_HTTP_TIMEOUT=60
	```
	"""
	def __init__(self) -> None:
		pass

	def add(package:str):
		# 安装开发和生产环境
		# This method seems to have incorrect logic based on the original code.
		# It should likely take a group argument or have separate methods.
		# Returning a generic command for now.
		return f"uv add {package}"

	def remove(package:str):
		return f"uv remove {package}"

	def init(project: str = "."):
		return f"uv init {project}"

	def build(self):
		return "uv build"

	def sync(self):
		return "uv sync"

	def run(script_path: str):
		return f"uv run {script_path}"

	def export_requirements(output_file: str = "requirements.txt"):
		return f"uv export --format requirements-txt > {output_file}"

	def python_dir(self):
		return "uv python dir"

	def tool_dir(self):
		return "uv tool dir"

	def python_list(self):
		return "uv python list"

	def python_install(version: str):
		return f"uv python install {version}"

	def upgrade_package(package: str):
		return f"uv lock --upgrade-package {package}"

	# Environment variables are typically set outside the script,
	# but we can provide methods to generate the commands.
	def set_cache_dir(path: str):
		return f"export UV_CACHE_DIR={path}"

	def set_index_url(url: str):
		return f"export UV_INDEX_URL={url}"

	def set_extra_index_url(url: str):
		return f"export EXTRA_INDEX_URL={url}"

	def set_no_cache(value: int = 0):
		return f"export UV_NO_CACHE={value}"

	def set_http_timeout(seconds: int = 60):
		return f"UV_HTTP_TIMEOUT={seconds}"


class docker_compose():
	"""
	Helper class for generating docker-compose commands and providing documentation.

	File layout example:
	```
	project
	- docker-compose.yml
	- Dockerfile.kimi
	- Dockerfile.obsidianrag
	```

	Ports mapping:
	Host:Container

	Restart policies:
	- `no`: Do not automatically restart the container (default).
	- `always`: Always automatically restart.
	- `on-failure`: Only restart if the container exits with a non-zero exit code.
	- `unless-stopped`: Always restart unless the container was explicitly stopped.
	"""
	def __init__(self) -> None:
		pass

	def build(self, service_name: str = None, force_rm: bool = False, no_cache: bool = False) -> str:
		"""
		Generates the docker-compose build command.

		Args:
			service_name: The name of the service to build. If None, builds all services.
			force_rm: Force removal of intermediate containers.
			no_cache: Do not use cache when building the image.
		"""
		command = "docker-compose build"
		if force_rm:
			command += " --force-rm"
		if no_cache:
			command += " --no-cache"
		if service_name:
			command += f" {service_name}"
		return command

	def up(self, service_name: str = None, build: bool = False, detach: bool = False, no_deps: bool = False) -> str:
		"""
		Generates the docker-compose up command.

		Args:
			service_name: The name of the service to start. If None, starts all services.
			build: Build images before starting containers.
			detach: Run containers in the background.
			no_deps: Don't start linked services.
		"""
		command = "docker-compose up"
		if build:
			command += " --build"
		if detach:
			command += " -d"
		if no_deps:
			command += " --no-deps"
		if service_name:
			command += f" {service_name}"
		return command

	def docker_run(self, image: str, command: str) -> str:
		"""
		Generates a basic docker run command.

		Args:
			image: The Docker image to run.
			command: The command to run inside the container.
		"""
		return f"docker run -i -t {image} {command}"

	def write_example_compose_file(self) -> str:
		"""
		Provides an example docker-compose.yml file content.
		"""
		return """
version: '3.8'

services:
  obsidianrag:
	restart: always
    build:
      context: .
      dockerfile: Dockerfile.obsidianrag
    ports:
      - "9000:9000"
    volumes:
      - .:/app
      - /Users/zhaoxuefeng/本地文稿/百度空间/cloud/Obsidian/知识体系尝试:/Users/zhaoxuefeng/本地文稿/百度空间/cloud/Obsidian/知识体系尝试
      - /Users/zhaoxuefeng/GitHub/test1:/Users/zhaoxuefeng/GitHub/test1
  kimi:
    build:
      context: .
      dockerfile: Dockerfile.kimi
    ports:
      - "9001:9000" # 主机:容器
    volumes:
      - .:/app
      - /Users/zhaoxuefeng/本地文稿/百度空间/cloud/Obsidian/知识体系尝试:/Users/zhaoxuefeng/本地文稿/百度空间/cloud/Obsidian/知识体系尝试
      - /Users/zhaoxuefeng/GitHub/test1:/Users/zhaoxuefeng/GitHub/test1
"""


class IPManager():
    """
    Helper class for generating commands related to network and process information.
    """
    def __init__(self) -> None:
        pass

    def find_process_by_port_netstat(self, port: int) -> str:
        """
        Generates a command to find processes listening on a specific port using netstat.

        Args:
            port: The port number to check.
        """
        return f"""sudo netstat -tunlp | grep "{port}" """

    def find_process_by_port_lsof(self, port: int) -> str:
        """
        Generates a command to find processes listening on a specific port using lsof.

        Args:
            port: The port number to check.
        """
        return f"""sudo lsof -i :{port} """

    def make_executable(self, script_path: str) -> str:
        """
        Generates a command to make a script executable.

        Args:
            script_path: The path to the script.
        """
        return f"chmod +x {script_path}"

    def move_script_to_bin(self, script_path: str, destination_dir: str = "/usr/local/bin") -> str:
        """
        Generates a command to move a script to a directory in the system's PATH.

        Args:
            script_path: The path to the script.
            destination_dir: The destination directory (defaults to /usr/local/bin).
        """
        return f"sudo mv {script_path} {destination_dir}"

    def run_python_script_from_string(self, python_code: str) -> str:
        """
        Generates a command to run Python code provided as a string.

        Args:
            python_code: The Python code to execute.
        """
        # Note: This is a basic example and might need escaping for complex code.
        return f"""echo "{python_code.replace('"', '\\"')}" | python3"""



class PiManager():
	"""
	Helper class for generating commands related to pip and private package indexes.
	"""
	def __init__(self):
		pass

	def update_private_index(self, whl_path: str, index_dir: str) -> str:
		"""
		Generates commands to copy a wheel file and update a private pip index.

		Args:
			whl_path: Path to the wheel file.
			index_dir: Directory of the private pip index.
		"""
		copy_command = f"cp {whl_path} {index_dir}"
		update_command = f"dir2pi {index_dir}"
		return f"{copy_command} && {update_command}"

	def install_from_private_index(self, package: str, index_url: str, index_dir: str) -> str:
		"""
		Generates a command to install a package from a private pip index.

		Args:
			package: The package name to install.
			index_url: The URL of the private index.
			index_dir: The directory of the private index (used by pip2pi).
		"""
		# Note: pip2pi is used to generate the index, pip install is used to consume it.
		# The index_dir is not directly used in the pip install command, but is needed for context.
		return f"pip install -i {index_url} {package}"

	def install_from_mirror(self, package: str, mirror_url: str) -> str:
		"""
		Generates a command to install a package from a specified mirror.

		Args:
			package: The package name to install.
			mirror_url: The URL of the mirror.
		"""
		return f"pip install -i {mirror_url} {package}"



import numpy as np
from IPython.display import display, Audio


class IpythonManager():
	"""
	Helper class for IPython related functionalities.
	"""
	def __init__(self) -> None:
		pass

	def play_tone(self, duration_seconds: int = 5, freq1: int = 220, freq2: int = 224, framerate: int = 44100):
		"""
		Plays a simple dual-tone sound in IPython.

		Args:
			duration_seconds: Duration of the tone in seconds.
			freq1: Frequency of the first tone.
			freq2: Frequency of the second tone.
			framerate: The sample rate.
		"""
		t = np.linspace(0, duration_seconds, framerate * duration_seconds)
		dataleft = np.sin(2 * np.pi * freq1 * t)
		dataright = np.sin(2 * np.pi * freq2 * t)
		display(Audio([dataleft, dataright], rate=framerate))

	def play_speech(self, data):
		"""
		Plays audio data in IPython.

		Args:
			data: The audio data to play.
		"""
		display(Audio(data, autoplay=True))



class Colab(Know):

    def 防止自动退出(self):
        return """
防止colab 自动退出
```python
#@title 1. Keep this tab alive to prevent Colab from disconnecting you { display-mode: "form" }
#@markdown Press play on the music player that will appear below:
%%html
<audio src="https://oobabooga.github.io/silence.m4a" controls> 用来防止发生断裂
```

"""




   # 适配器


   def 防止自动退出():
      #%%html
      #<audio src="https://oobabooga.github.io/silence.m4a" controls> 用来防止发生断裂
      pass

   # 挂载云盘
   def drive():
      from google.colab import drive
      drive.mount('/content/drive')



   # 环境变量
   def get_veriable():
      from google.colab import userdata
      api_key = userdata.get('api_key')




class Xmind:
	def work1():

		Xmind 导出为opml格式
		```
		- 直接将<head> ... </head>删除
		```


		mindly 导出为txt 将txt 复制粘贴到Xmind中



#### GitHub


class Githubs():
	def __init__(self) -> None:
		pass

	def multi_user_manage(self):
		# GitHub Desktop 多账户管理 文件 > 选项 > 账户





class ServerMechine():
	def __init__(self) -> None:
		pass

	def set_server(self):
		# - 关闭自动睡眠   - 设置-> 锁定屏幕 -> 不活跃时启动屏幕保护程序 -> 永不,永不 永不
		# - 开机自动登录   - 用户和群组 -> 自动以此身份登录
		# - 断点后自动开机 - 节能 -> 断电后自动开机
		# - 远程访问      - 设置-> 通用-> 共享 -> 远程管理

	def build_mechine(self):
		pass

	def get_info_mechine(self):
		"""
		uname -m #查看架构
		df -h # 查看磁盘
		lscpu # cpu信息
		nvidia-smi # gpu信息
		ping -4 www.baidu.com
		"""

	def create_user(self):
		"""
		创建用户组
		```
		adduser username
		passwd username
		usermod -aG sudo qework
		```
		"""

	def init_server(self):
		# 安装基础依赖包
		"""
		
		sudo apt update
		sudo apt install build-essential
		apt install python3
		apt install vim git curl htop -y

		"""

	def anz_conda(self):
		"""
		安装conda
		wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

		bash Miniconda3-latest-Linux-x86_64.sh

		"""


class XClashh():
	def 如何在局域网中设置代理():
		"""
## 可以让设备访问外网

## 方法
windows10 -> 设置 -> 代理 -> 启用“使用代理服务器” -> 输入代理服务器地址和端口号
mac -> 点击当前连接的Wi-Fi网络旁边的“i”按钮 -> 配置代理”->“手动” -> 输入代理服务器地址和端口号
Oculurs -> 网络 -> 高级设置 -> 代理 ->输入代理服务器地址和端口号


如何使用ClashX控制操作


## 性质
**全局模式**：所有流量都通过代理节点，适合需要全面保护的情况。
**规则模式**：根据配置文件中的规则，决定哪些流量走代理，哪些直连。
**直连模式**：所有流量不走代理，适合访问本地网络。
**脚本模式（Script Mode）**是一种高级功能，允许用户通过自定义脚本来动态处理网络请求的分流

**设置系统代理**：如果希望所有应用都通过ClashX代理，可以在ClashX的菜单栏图标中选择“设置为系统代理”。

可以使用局域网共享来分享VPN

## 方法
分享 VPN : **允许局域网连接**
记下显示的HTTP和SOCKS代理端口，通常默认HTTP端口为7890。

1. 打开设备的Wi-Fi设置，点击当前连接的Wi-Fi网络旁边的“i”按钮。
2. 滑动到底部，选择“配置代理”->“手动”。
3. 填写以下信息：
4. - **服务器**：输入ClashX运行的电脑的IP地址（如`192.168.x.x`）。
        
    - **端口**：输入ClashX的HTTP代理端口（通常是`7890`）。
        
    - **认证**：通常不开启。
        
设置[[如何在局域网中设置代理]]: 分享本机的外网网络给局域网的其他设备

将配置移到新的设备中: 配置 -> 打开配置文件夹 -> 将对应的配置文件夹移动到新位置


查看端口被占用情况


		"""



project_name=memorier

cp ../tools/mkdocs.yml .
mkdir $project_name && touch $project_name/core.py
mkdir test && touch test/test_.py
uv init .
uv sync
rm main.py   
mkdocs new .
bash update_docs.sh 
mkdocs gh-deploy -d ../.temp
bash run_build.sh 
uv publish



"""

# 初始化一个仓库的步骤

使用该方式 仓库 来固定知识


0 起名 查看pypi 是否重名
1 在能力面板创建其内容
2 创建仓库
3 克隆

0 uv init .
1 uv sync


1 mkdocs new .
2 p_pulldocs 
3 构建包 init
4 更新readme  pyproject
5 更新mkdocs.yml  
6 mkdocs gh-deploy -d ../.temp
7 p_pushdocs 
8 p_build
9 uv publish

测试的时候
1 在jupyter 上使用 use case
2 写入use case
3 编写细节的 pytesr
测试完再上的应该