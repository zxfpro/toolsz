"""
深度学习训练和评价工具
author：zhaoxuefeng
datetime：2021-09-20 16:33:06
"""
import torch
from d2l import torch as d2l



def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().
        torch.device('cpu')
        torch.device('cuda')
        torch.cuda.device_count()

    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]



def show_layers_size(net,X_size=(1, 1, 28, 28)):
    X = torch.rand(size=X_size, dtype=torch.float32)
    for layer in net:
        X = layer(X)
        yield layer.__class__.__name__, 'output shape: \t', X.shape



def train_batch_(net,X,y,loss,train_optim,devices):
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    train_optim.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    train_optim.step()
    train_loss_sum = l.sum()

    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum



import time
import numpy as np

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()



import pytest

@pytest.fixture()
def user():
    return 'admin', '123456'

@pytest.fixture
def login(user):  # Fixture函数可以引用(依赖)其他Fixture函数
    # 使用的user参数实际为调用user()函数后的返回值
    username, password = user
    print(f'{username} 登录')
    token = 'abcdef'
    yield token  # yield上为测试准备，yield下为测试清理
    print('退出登录’)


# 翻译英语
import re,os




def to_translate_Chinese(word):
    #pip install quicktranslate
    with open('word_of_mine.txt','a') as a:
        a.write(word+'\n')
    os.system('trans -t %s>word.txt'%word)
    os.system('trans -t %s>>word_of_mine.txt'%word)
    with open('word.txt','r') as r:
        content = r.read()
    first_processing = re.sub(r"youdao translate result： ", '/', content)
    second_processing = re.sub(r"\nbaidu translate result：", '/', first_processing)
    last_processing = second_processing.split('/')[1]
    result = re.sub(r' ', '_', last_processing).lower()
    return result


if __name__ == '__main__':
    print(to_translate_Chinese('去购物'))


    