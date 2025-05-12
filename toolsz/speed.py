"""
符号式编程方式工具

"""
import torch
import torch.nn as nn

def transjit(net):
    return torch.jit.script(net)

# ppkl = """
# import torch
# def add(a, b):
#     return a + b
#
# def fancy_func(a, b, c, d):
#     e = add(a, b)
#     f = add(c, d)
#     g = add(e, f)
#     return g
# print(fancy_func(1, 2, 12, 4))
# """
# z = compile(ppkl,'','exec')
# exec(z)
#



'''
1命令式编程（解释型）
程序效率不高，没有考虑到复用情况
调试更加简单
2符号是编程

符号式编程运行效率更高，更易于移植。
符号式编程更容易在编译期间优化代码，同时还能够将程序移植到与 Python 无关的格式中，
从而允许程序在非 Python 环境中运行，
避免了任何潜在的与 Python 解释器相关的性能问题。
'''