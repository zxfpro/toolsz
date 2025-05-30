
from loguru import logger


# setting
# 添加一个文件输出，记录 DEBUG 及以上级别，按文件大小分割，保留最新 7 个文件
logger.add("my_app.log", level="DEBUG", rotation="1 MB", retention="7 days", compression="zip")


# session_id

# logger添加
logger.debug("这是调试信息")
logger.info("这是信息")
logger.warning("这是警告")
logger.error("这是错误")
logger.critical("这是严重错误")


# catch

@logger.catch # 装饰器用法

with logger.catch(): # 上下文管理器用法
    pass


# bind
user_logger = logger.bind(user_id="abc", session_id="xyz")
user_logger.info("User session started")
# 输出可能包含 user_id="abc", session_id="xyz"


###########



```python
from loguru import logger
import os
from datetime import datetime

# 定义日志文件路径
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_{datetime.now().strftime('%Y%m%d')}.log")

# 配置日志记录器
logger.configure(
    handlers=[
        # 输出到控制台
        {"sink": "stdout", "format": "{time} - {level} - {message}", "level": "DEBUG"},
        # 输出到文件
        {"sink": log_file, "format": "{time} - {level} - {message}", "level": "DEBUG", "rotation": "500 MB", "retention": "7 days"},
    ]
)

# 添加一个过滤器，可以根据需要过滤日志
def filter_logs(record):
    # 只记录 INFO 及以上级别的日志
    return record["level"].no >= logger.level("INFO").no

# 添加一个自定义的日志级别
logger.level("CUSTOM", no=38, color="<yellow>", icon="⚡")

# 使用自定义日志级别
logger.log("CUSTOM", "This is a custom level message")

# 记录不同级别的日志
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.success("This is a success message")
logger.warning("This is a warning message")
logger.error("This is an error message")
logger.critical("This is a critical message")

# 使用上下文日志记录
with logger.contextualize(user_id=123, request_id="abc"):
    logger.info("This log has context: {user_id}, {request_id}")

# 捕获异常并记录堆栈信息
try:
    1 / 0
except ZeroDivisionError as e:
    logger.error("An error occurred: {}", e)
    logger.opt(exception=e).error("An error occurred with traceback")

# 异步记录日志
async def async_function():
    logger.info("This is an async log message")

# 使用自定义格式化函数
def format_log(record):
    # 自定义格式化函数，可以修改记录的内容
    record["extra"]["custom_time"] = record["time"].strftime("%Y-%m-%d %H:%M:%S")
    return "{extra[custom_time]} - {level} - {message}\n"

logger.remove()  # 移除默认的处理程序
logger.add("formatted_log.log", format=format_log, level="DEBUG")

logger.info("This log uses a custom format")
```