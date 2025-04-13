""" 时间小工具 """
from datetime import datetime

def get_today():
    """返回当前本地日期，格式为 'YYYY-MM-DD' 的字符串。

    此函数使用系统的本地时间来确定当前日期。

    返回:
        str: 当前本地日期，格式为 'YYYY-MM-DD'。
    """
    local_time = datetime.today()
    local_time = local_time.strftime("%Y-%m-%d")
    # local_time = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return local_time
