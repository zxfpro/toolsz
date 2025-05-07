import os
import functools
import qrcode as qrcode_
import pickle

class PythonLib:
    def __init__(self):
        pass

    @staticmethod
    def listdir():
        return os.listdir()

    @staticmethod
    def getenv(key, default=None):
        return os.environ.get(key, default)

    @staticmethod
    def setenv(key, value):
        os.environ[key] = value

    @staticmethod
    def save_pkl(x):
        temp_path = os.environ.get("pickle_temp_path")
        temp_file = os.path.join(temp_path, "data.pkl")
        with open(temp_file, "wb") as file:
            pickle.dump(x, file)

    @staticmethod
    def load_pkl():
        temp_path = os.environ.get("pickle_temp_path")
        temp_file = os.path.join(temp_path, "data.pkl")
        with open(temp_file, "rb") as file:
            loaded_data = pickle.load(file)
        return loaded_data

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