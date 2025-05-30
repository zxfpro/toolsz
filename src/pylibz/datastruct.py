# 算法

```python
import pandas as pd
import numpy as np

xx = pd.DataFrame(np.zeros((9,9)))

a = [0,1,2,3,4,5,6,7,8]

# 遍历所有的格子
for i in range(len(a)):
    for j in range(len(a)):
        xx.iloc[i][j] = 2
```

```python
xx = pd.DataFrame(np.zeros((9,9)))

# 遍历所有的格子
for i in range(len(a)):
    for j in range(i,len(a)):
        xx.iloc[i][j] = 2
```


```python
xx = pd.DataFrame(np.zeros((9,9)))

# 遍历所有的格子
for i in range(len(a)):
    for j in range(i):
        xx.iloc[i][j] = 2
```

```python
xx = pd.DataFrame(np.zeros((9,9)))

# 遍历所有的格子
for i in range(len(a)):
    for j in range(len(a)):
        if i == j:
            xx.iloc[i][j] = 2
```

```python

class Core:
    def __init__(self,left,right):
        self.left = left
        self.right = right
        
    def __lt__(self,other):
        print('小于')
        return True
    
    def __gt__(self,other):
        print('大于')
        return False
    
    def __eq__(self,other):
        print('等于')
        return True


a = Core(1,2)
b = Core(2,3)

a==b

a>b

```

```python
def binary_search(nums: list[int], target: int) -> int:
    """二分查找（双闭区间）"""
    # 初始化双闭区间 [0, n-1] ，即 i, j 分别指向数组首元素、尾元素
    i, j = 0, len(nums) - 1
    # 循环，当搜索区间为空时跳出（当 i > j 时为空）
    while i <= j:
        # 理论上 Python 的数字可以无限大（取决于内存大小），无须考虑大数越界问题
        m = (i + j) // 2  # 计算中点索引 m
        if nums[m] < target:
            i = m + 1  # 此情况说明 target 在区间 [m+1, j] 中
        elif nums[m] > target:
            j = m - 1  # 此情况说明 target 在区间 [i, m-1] 中
        else:
            return m  # 找到目标元素，返回其索引
    return -1  # 未找到目标元素，返回 -1

```


插入排序

折半插入排序

希尔排序




概念
维护一个窗口  有左右两个指针
扩大和缩小窗口


性质
基本用于数组或字符串问题, 解决子数组或子串相关的问题
通常可以将时间复杂度从 n方 优化到 n



边的添加和删除

顶点的添加和删除

初始化



[[数据结构-有哪些排序算法]]

[[插入排序的内容]]

[[数据结构-冒泡排序]]

[[数据结构-选择排序]]

[[数据结构-堆排序]]

[[数据结构-归并排序]]

[[数据结构-基数排序]]

[[数据结构-解决实际问题]]





```python

# 线性表

from abc import ABC, abstractmethod

ssss测试


class ADT(ABC):
    def __init__(self):
        pass

    def Destory(self):
        # 销毁
        pass

    def Clear(self):
        # 重置
        pass

    @staticmethod
    def Empty(L)->bool:
        # 条件: 线性表已存在
        # 判断空
        pass

    @staticmethod
    def Length(L)->int:
        pass

    @staticmethod
    def Traverse(L,visit):
        pass


# 顺表
class List(ADT):
    def __init__(self):
        self.value = None

    def Create(self,value):
        self.value = value

    @staticmethod
    def GetElem(L,i)->int:
        # 条件: 线性表已存在 且 1<=i<=length
        # 返回第i个元素
        assert 1<=i<=List.ListLength(L)

        return L.value[i-1]

    def ListInsert(self,i,e):
        # 条件: 线性表已存在 且 1<=i<=length+1
        # 在第i 个位置之前插入数据 L的长度+1
        self.value.insert(i-1,e)

    def ListDelete(self,i)->'e':
        # 条件: 线性表已存在
        # 删除第i个元素数据 长度减1
        self.value.pop(i-1)

    @staticmethod
    def ListTraverse(L,visit=None):
        # 条件: 线性表已存在
        for i in L.value:
            if visit:
                visit(i)

    # 查找算法
    @staticmethod
    def search(lst,key):
        lst2 = [key] + lst
        for i in range(len(lst2)-1,-1,-1):
            if lst2[i] == key:
                return i
    @staticmethod
    def search_bin(lst,key):
        # 折半查找
        low = 1
        high = len(lst)

        while low <=high:
            mid = (low+high)//2
            if key == lst[mid-1]:
                return mid
            elif key < lst[mid-1]:
                high = mid -1
            else:
                low= mid+1
        return 0

    # 8大排序
    def insert_sort(alist):
        # 从第二个位置，即下标为1的元素开始向前插入
        for i in range(1, len(alist)):
            # 从第i个元素开始向前比较，如果小于前一个元素，交换位置
            for j in range(i, 0, -1):
                if alist[j] < alist[j-1]:
                    alist[j], alist[j-1] = alist[j-1], alist[j]
    def bin_sort(alist):
        for i in range(1, len(alist)):
            key = alist[i]
            low = 0
            high = i - 1
            while low <= high:  # 二分查找确定插入位置
                mid = (low + high) // 2
                if key < alist[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            for j in range(i, low, -1):  # 移动元素腾出空间
                alist[j] = alist[j - 1]
            alist[low] = key  # 插入元素
        return alist

    # 冒泡排序

    def bubble_sort(alist):
        n = len(alist)
        for i in range(n):
            for j in range(0, n-i-1):
                if alist[j] > alist[j+1]:
                    alist[j], alist[j+1] = alist[j+1], alist[j]
        return alist


    def quicksort(alist):
        # 快速排序
        def partition(arr, low, high):
            i = low - 1
            pivot = arr[high]

            for j in range(low, high):
                if arr[j] < pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]

            arr[i+1], arr[high] = arr[high], arr[i+1]
            return i+1

        def quick_sort(arr, low, high):
            if low < high:
                pi = partition(arr, low, high)
                quick_sort(arr, low, pi-1)
                quick_sort(arr, pi+1, high)
            return arr
        quick_sort(alist,0,len(alist)-1)
    def selectionSort(alist):
        #选择排序
        for i in range(len(alist) - 1):
            # 记录最小数的索引
            minIndex = i
            for j in range(i + 1, len(alist)):
                if alist[j] < alist[minIndex]:
                    minIndex = j
            # i 不是最小数时，将 i 和最小数进行交换
            if i != minIndex:
                alist[i], alist[minIndex] = alist[minIndex], alist[i]
        return alist


    def heapsort(arr):
        # 堆排序
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2

            if left < n and arr[i] < arr[left]:
                largest = left

            if right < n and arr[largest] < arr[right]:
                largest = right

            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                heapify(arr, n, largest)

        def heap_sort(arr):
            n = len(arr)

            for i in range(n // 2 - 1, -1, -1):
                heapify(arr, n, i)

            for i in range(n-1, 0, -1):
                arr[i], arr[0] = arr[0], arr[i]
                heapify(arr, i, 0)
            return arr
        heap_sort(arr)

    def shellSort(arr):
        # 希尔排序
        n = len(arr)
        # 这里的gap取什么是根据你的实际状况取最优，为了展示方便，选择了取半
        gap = n // 2
        while gap > 0:
            for i in range(gap,n):
                j = i
                while j > 0:
                    # 到这里就很插入有不一样的地方了，对比的不是-1，而是减去步长
                    if arr[j] < arr[j-gap]:
                        arr[j], arr[j-gap] = arr[j-gap], arr[j]
                        # 这里也是同理
                        j -= gap
                    else:
                        break
            # 然后在进行分组 直到 while的gap不是为0为止
            gap //= 2
        return arr

    def radix_sort(arr):
        # 基数排序
        # 计算最大位数
        max_num = max(arr)
        digits = len(str(max_num))

        for i in range(digits):
            # 创建空桶
            buckets = [[] for _ in range(10)]

            # 将元素分配到桶中
            for num in arr:
                index = (num // 10**i) % 10
                buckets[index].append(num)

            # 合并桶
            arr = []
            for bucket in buckets:
                arr.extend(bucket)

        return arr

    def mergesort(arr):
        # 归并排序
        def merge(left,right):
            result = []
            while left and right:
                if left[0] <= right[0]:
                    result.append(left.pop(0))
                else:
                    result.append(right.pop(0));
            while left:
                result.append(left.pop(0))
            while right:
                result.append(right.pop(0));
            return result
        def mergeSort(arr):
            import math
            if(len(arr)<2):
                return arr
            middle = math.floor(len(arr)/2)
            left, right = arr[0:middle], arr[middle:]
            return merge(mergeSort(left), mergeSort(right))

        return mergeSort(arr)


# 链
class Node:
    def __init__(self,data):
        self.data = data
        self.next = None

class DNode:
    def __init__(self, data):
        self.data = data  # 节点存储的数据
        self.prev = None
        self.next = None


# 链表
class LinkList(ADT):
    def __init__(self):
        self.head = Node(0)

    def CreateHead(self,value):
        for i in value:
            p = Node(i)
            p.next = self.head.next
            self.head.next = p

            self.head.data +=1

    def CreateTail(self,value):
        p = self.head
        for i in value:
            p.next = Node(i)
            p = p.next

            self.head.data +=1

    def _getNode(self,i)->Node:
        p = self.head.next
        j = 1
        while j<i-1:
            p = p.next
            j+=1
        return p

    @staticmethod
    def GetElem(L,i)->int:
        # 条件: 线性表已存在 且 1<=i<=length
        # 返回第i个元素
        assert 1<=i<=L.head.data
        p = L._getNode(i+1)
        return p.data

    def ListInsert(self,i,e):
        # 条件: 线性表已存在 且 1<=i<=length+1
        # 在第i 个位置之前插入数据 L的长度+1
        assert 1<=i<=self.head.data+1
        p = self._getNode(i)
        new_node = Node(e)
        new_node.next = p.next
        p.next = new_node
        self.head.data +=1

    def ListDelete(self,i)->'e':
        # 条件: 线性表已存在
        # 删除第i个元素数据 长度减1
        assert 1<=i<=self.head.data
        p = self._getNode(i)
        p.next = p.next.next
        self.head.data -=1

    @staticmethod
    def ListTraverse(L,visit=None):
        # 条件: 线性表已存在
        p = L.head
        while p.next:
            p = p.next
            if visit:
                visit(p.data)

# 循环链表
class CircularLinkList(LinkList):
    def __init__(self):
        self.head = Node(0)
        self.head.next = self.head

    def CreateTail(self,value):
        p = self.head
        for i in value:
            p.next = Node(i)
            p = p.next
            p.next = self.head #// 有链表新增 //
            self.head.data +=1

    @staticmethod
    def ListTraverse(L,visit=None):
        # 条件: 线性表已存在
        p = L.head
        while p.next!=L.head:# TODO p.next  解决循环尾端的问题
            p = p.next
            if visit:
                visit(p.data)

# 双向链表
class DoublyLinkList(LinkList):
    def __init__(self):
        self.head = DNode(0)

    def CreateHead(self,value):
        for i in value:
            p = DNode(i)
            p.prov = self.head
            p.next = self.head.next

            self.head.next = p

            if p.next:          # TODO
                p.next.prov = p # TODO
            self.head.data +=1

    def CreateTail(self,value):
        p = self.head
        for i in value:
            p.next = DNode(i)
            p.next.prov = p # TODO
            p = p.next
            self.head.data +=1

    def ListInsert(self,i,e):
        # 条件: 线性表已存在 且 1<=i<=length+1
        # 在第i 个位置之前插入数据 L的长度+1
        assert 1<=i<=self.head.data+1
        p = self._getNode(i)
        new_node = DNode(e) # TODO
        new_node.prov = p  # TODO
        new_node.next = p.next
        p.next = new_node
        new_node.next.prov = new_node # TODO

        self.head.data +=1

    def ListDelete(self,i)->'e':
        # 条件: 线性表已存在
        # 删除第i个元素数据 长度减1
        assert 1<=i<=self.head.data
        p = self._getNode(i)
        p.next = p.next.next
        p.next.prov = p      # TODO
        self.head.data -=1


# 栈和队列

# 顺序栈

class Stack(ADT):
    def __init__(self,stacksize = 10):
        # 初始化一个空栈

        self.store = [None]*stacksize
        self.stacksize = stacksize
        self.top = 0
        self.base = 0

    def Create(self,value):
        for i in value:
            self.Push(i)

    @staticmethod
    def GetTop(S)->'data':
        # 栈存在且非空
        # 查看栈顶元素
        assert not S.top == S.base
        e = S.store[S.top -1]
        return e

    def Push(self,e):
        # 栈存在
        # 压栈
        if self.top - self.base >=self.stacksize:
            # 扩容
            self.store += [None]*10
            self.stacksize +=10

        self.store[self.top] = e
        self.top +=1


    def Pop(self)->'data':
        # 栈存在
        # 弹出栈顶元素
        assert not self.top == self.base
        self.top -=1
        e = self.store[self.top]
        return e

    @staticmethod
    def StackTraverse(S,visit):
        # 遍历
        for i in S.store[:S.top]:
            if visit:
                visit(i)


# 循环队列
class CriQueue(ADT):
    def __init__(self,queuesize = 10):
        # 初始化一个空栈

        self.store = [None]*queuesize
        self.queuesize = queuesize
        self.front = 0
        self.rear = 0

    @staticmethod
    def GetHead(Q)->'data':
        # 存在且非空
        # 查看队头元素
        assert not Q.rear == Q.front
        e = Q.store[Q.front]
        return e

    @staticmethod
    def QueueLength(Q):
        return (Q.rear - Q.front + Q.queuesize)% Q.queuesize


    def EnQueue(self,e):
        # 存在
        # 插入队尾元素
        print(self.rear,self.front,self.queuesize)
        assert (self.rear + 1) % self.queuesize != self.front
        self.store[self.rear] = e
        self.rear = (self.rear+1) % self.queuesize


    def DeQueue(self)->'data':
        # 存在
        # 删除队头元素
        assert not self.front == self.rear
        e = self.store[self.front]
        self.front = (self.front+1) % self.queuesize
        return e

    @staticmethod
    def StackTraverse(S,visit):
        # 遍历
        for i in S.store[:S.top]:
            if visit:
                visit(i)


# 链队列
class LinkQueue(LinkList):
    def __init__(self):
        # 初始化一个空队列
        self.front = self.rear = Node(0)
        self.head = self.front

    @staticmethod
    def GetHead(Q)->'data':
        e = Q.head.next.data
        return e

    def EnQueue(self,e):
        # 存在
        # 插入队尾元素
        new_node = Node(e)
        self.rear.next = new_node
        self.rear = new_node
        self.head.data +=1


    def DeQueue(self)->'data':
        # 存在
        # 删除队头元素
        assert not self.front == self.rear
        p = self.front.next
        e = p.data
        self.front.next = p.next
        self.front.data -= 1
        if self.rear == p:
            self.rear = self.front
        return e


def Index(S,T):
    # 模式匹配 暴力匹配
    n = len(S)
    m = len(T)
    if m == 0:
        return 0
    i = 1
    while i <(n-m+1):
        for j in range(m):
            if S[i+j] != T[j]:
                # print('失败')
                break
        else:
            # print('匹配成功')
            return i
        i+=1
    return 0





%run 线性表.ipynb



# 树


# node

class BSTNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

class Tree(ADT):
    def __init__(self):
        pass

class BinTree(Tree):
    def __init__(self):
        # 示例使用
        # 构建二叉树
        #       2
        #      / \
        #     1   4
        #    / \  / \
        #   3  2 5  0
        #  /
        #  9
        self.root = BSTNode(2)

    def CreateBiTree(self):
        self.root.left = BSTNode(1)
        self.root.right = BSTNode(4)
        self.root.left.left = BSTNode(3)
        self.root.left.right = BSTNode(2)
        self.root.right.left = BSTNode(5)
        self.root.right.right = BSTNode(0)
        self.root.left.left.left = BSTNode(9)

    @staticmethod
    def PreOrderTraverse(T,visit=None):
        # TODO 考虑线索化
        if T:
            visit(T.data)
            BinTree.PreOrderTraverse(T.left,visit)
            BinTree.PreOrderTraverse(T.right,visit)

    @staticmethod
    def InOrderTraverse(T,visit=None):
        # TODO 考虑线索化
        if T:
            BinTree.PreOrderTraverse(T.left,visit)
            visit(T.data)
            BinTree.PreOrderTraverse(T.right,visit)


    @staticmethod
    def PostOrderTraverse(T,visit=None):
        # TODO 考虑线索化
        if T:
            BinTree.PreOrderTraverse(T.left,visit)
            BinTree.PreOrderTraverse(T.right,visit)
            visit(T.data)

    @staticmethod
    def LevelOrderTraverse(T,visit):
        # TODO 考虑线索化
        pass





# 图

图
	CreateGraph(\&G,V,VR);
		初始条件： V 是图的顶点集， VR 是图中弧的集合。
		操作结果：按 V和VR的定义构造图G。
	LocateVex (G, U)；
		初始条件：图 G 存在， u 和 G 中顶点有相同特征。
		操作结果：若 G 中存在顶点 u ，则返回该顶点在图中位置；否则返回其他信息。
	GetVex(G, v);
		初始条件：图 G存在， V 是 G 中某个顶点。
		操作结果：返回 V 的值。
	DFSTraverse(G, Visit())；
		初始条件：图 G 存在，Visit 是顶点的应用函数。
		操作结果：对图进行深度优先遍历。在遍历过程中对每个顶点调用函数 Visit 一次且仅一次。一旦 visit（）失败，则操作失败。
		深度优先搜索
		使用栈
	BFSTraverse(G, Visit())；
		初始条件：图G存在, Visit 是顶点的应用函数。
		操作结果：对图进行广度优先遍历。在遍历过程中对每个顶点调用函数 Visit 一次且仅一次。一旦 visit(）失败，则操作失败。
		广度优先搜索
		做一个标志用来看节点是否被访问过
		使用队列
```



VOE网

- AOV
    
    - 定义
        
        - 在这种有向图中,顶点表示活动,边表示活动的优先关系 Action on Vertices
            
        - 直接前驱,直接后继
            
        - 反自反性
            
        - 不能出现环
            
    - 对规定的AOV网
        
        - 要先判断它是否有环, 方法是对它进行拓扑排序, 将其顶点排列成线性有序序列, 若该序列中包含全部顶点, 则无环,
            
    - 拓扑排序的步骤
        
        - 在AOV网中选择一个入度为0的顶点且输出
            
        - 从AOV网中删除该顶点和该顶点发出的所有有向边
            
        - 重复, 直到AOV网中所有顶点都被输出或网中不存在入度为0的顶点








https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/#summary


https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial11/NF_image_modeling.html#


```python

import numpy as np
def sort_function_check(func):
    for i in range(1000):
        datas = np.random.randint(1,10,(100)).tolist()
        answer = np.sort(datas).tolist()
        our_answer = func(datas)
        assert answer == our_answer
    return 'pass check'

# 冒泡排序

def bubble_sort(lst):
    n = len(lst)
    for i in range(n):
        for j in range(1, n - i):
            if lst[j - 1] > lst[j]:
                lst[j - 1], lst[j] = lst[j], lst[j - 1]
    return lst

%%time
sort_function_check(bubble_sort)

# 选择排序

def selection_sort(lst):
    for i in range(len(lst) - 1):
        min_index = i
        for j in range(i + 1, len(lst)):
            if lst[j] < lst[min_index]:
                min_index = j
        lst[i], lst[min_index] = lst[min_index], lst[i]
    return lst

%%time
sort_function_check(selection_sort)

# 快速排序

def quick_sort(lst):
    n = len(lst)
    if n <= 1:
        return lst
    baseline = lst[0]
    left = [lst[i] for i in range(1, len(lst)) if lst[i] < baseline]
    right = [lst[i] for i in range(1, len(lst)) if lst[i] >= baseline]
    return quick_sort(left) + [baseline] + quick_sort(right)

%%time
sort_function_check(quick_sort)


# 归并排序

def merge_sort(lst):
    def merge(left,right):
        i = 0
        j = 0
        result = []
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result = result + left[i:] + right[j:]
        return result
    n = len(lst)
    if n <= 1:
        return lst
    mid = n // 2
    left = merge_sort(lst[:mid])
    right = merge_sort(lst[mid:])
    return merge(left,right)


%%time
sort_function_check(merge_sort)

# 堆排序

def heap_sort(lst):
    def adjust_heap(lst, i, size):
        left_index = 2 * i + 1
        right_index = 2 * i + 2
        largest_index = i
        if left_index < size and lst[left_index] > lst[largest_index]:
            largest_index = left_index
        if right_index < size and lst[right_index] > lst[largest_index]:
            largest_index = right_index
        if largest_index != i:
            lst[largest_index], lst[i] = lst[i], lst[largest_index]
            adjust_heap(lst, largest_index, size)

    def built_heap(lst, size):
        for i in range(len(lst )/ /2)[::-1]:
            adjust_heap(lst, i, size)

    size = len(lst)
    built_heap(lst, size)
    for i in range(len(lst))[::-1]:
        lst[0], lst[i] = lst[i], lst[0]
        adjust_heap(lst, 0, i)
    return lst


%%time
sort_function_check(heap_sort)

# 插入排序

def insertion_sort(lst):
    for i in range(len(lst) - 1):
        cur_num, pre_index = lst[i+1], i
        while pre_index >= 0 and cur_num < lst[pre_index]:
            lst[pre_index + 1] = lst[pre_index]
            pre_index -= 1
        lst[pre_index + 1] = cur_num
    return lst

%%time
sort_function_check(insertion_sort)

#希尔排序

def shell_sort(lst):
    n = len(lst)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            for j in range(i, gap - 1, -gap):
                if lst[j] < lst[j - gap]:
                    lst[j], lst[j - gap] = lst[j - gap], lst[j]
                else:
                    break
        gap //= 2
    return lst

%%time
sort_function_check(shell_sort)

# 计数排序

def counting_sort(lst):
    nums_min = min(lst)
    bucket = [0] * (max(lst) + 1 - nums_min)
    for num in lst:
        bucket[num - nums_min] += 1
    i = 0
    for j in range(len(bucket)):
        while bucket[j] > 0:
            lst[i] = j + nums_min
            bucket[j] -= 1
            i += 1
    return lst

%%time
sort_function_check(counting_sort)

#桶排序

def bucket_sort(lst, defaultBucketSize=4):
    maxVal, minVal = max(lst), min(lst)
    bucketSize = defaultBucketSize
    bucketCount = (maxVal - minVal) // bucketSize + 1
    buckets = [[] for i in range(bucketCount)]
    for num in lst:
        buckets[(num - minVal) // bucketSize].append(num)
    lst.clear()
    for bucket in buckets:
        bubble_sort(bucket)
        lst.extend(bucket)
    return lst


%%time
sort_function_check(bucket_sort)

# 基数排序


# LSD Radix Sort
def radix_sort(lst):
    mod = 10
    div = 1
    mostBit = len(str(max(lst)))
    buckets = [[] for row in range(mod)]
    while mostBit:
        for num in lst:
            buckets[num // div % mod].append(num)
        i = 0
        for bucket in buckets:
            while bucket:
                lst[i] = bucket.pop(0)
                i += 1
        div *= 10
        mostBit -= 1
    return lst

%%time
sort_function_check(bucket_sort)



# 栈
class Stack():
    def __init__(self):
        self.__list = []

    def is_empty(self):
        return self.__list == []

    def size(self):
        return len(self.__list)

    def push(self, item):
        self.__list.append(item)

    def pop(self):
        self.__list.pop()

    def peek(self):
        if self.__list:
            return self.__list[-1]
        else:
            return None

#队列
class Queue(object):
    """队列"""
    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        """进队列"""
        self.items.insert(0,item)

    def dequeue(self):
        """出队列"""
        return self.items.pop()

    def size(self):
        """返回大小"""
        return len(self.items)

#树
class Node(object):
    """节点类"""
    def __init__(self, elem=-1, lchild=None, rchild=None):
        self.elem = elem
        self.lchild = lchild
        self.rchild = rchild

class Tree(object):
    """树类"""
    def __init__(self, root=None):
        self.root = root

    def add(self, elem):
        """为树添加节点"""
        node = Node(elem)
        #如果树是空的，则对根节点赋值
        if self.root == None:
            self.root = node
        else:
            queue = []
            queue.append(self.root)
            #对已有的节点进行层次遍历
            while queue:
                #弹出队列的第一个元素
                cur = queue.pop(0)
                if cur.lchild == None:
                    cur.lchild = node
                    return
                elif cur.rchild == None:
                    cur.rchild = node
                    return
                else:
                    #如果左右子树都不为空，加入队列继续判断
                    queue.append(cur.lchild)
                    queue.append(cur.rchild)

    @classmethod
    def preorder(self, root):
        """递归实现先序遍历"""
        if root == None:
            return
        print(root.elem)
        self.preorder(root.lchild)
        self.preorder(root.rchild)

    @classmethod
    def inorder(self, root):
        """递归实现中序遍历"""
        if root == None:
            return
        self.inorder(root.lchild)
        print(root.elem)
        self.inorder(root.rchild)

    @classmethod
    def postorder(self, root):
        """递归实现后续遍历"""
        if root == None:
            return
        self.postorder(root.lchild)
        self.postorder(root.rchild)
        print(root.elem)

    @classmethod
    def breadth_travel(self, root):
        """利用队列实现树的层次遍历"""
        if root == None:
            return
        queue = []
        queue.append(root)
        while queue:
            node = queue.pop(0)
            print
            node.elem,
            if node.lchild != None:
                queue.append(node.lchild)
            if node.rchild != None:
                queue.append(node.rchild)

#递归
class To_factorial:
    """
    递归
    1 定义一个阶乘函数f(x)
    2 明确等级关系 5的阶乘 = 5 * 4的阶乘
    3 明确出口 1的阶乘 = 1
    4 写出函数
    def 阶乘(x):
        if x ==1:
            return 1
        else:
            return x*阶乘(x-1)
    """

    def __init__(self):
        pass

## 单链表
class SingleNode(object):
    def __init__(self, item):
        self.item = item
        self.next = None


class SingleLinkList(object):
    """单链表
    is_empty() 链表是否为空
    create() 创建
    length() 链表长度
    travel() 遍历链表
    add(item) 链表头部添加元素
    append(item) 链表尾部添加元素
    insert(pos, item) 指定位置添加元素
    remove(item) 删除节点
    find(item) 查找节点是否存在
    """

    def __init__(self):
        self._head = None

    def is_empty(self):
        """判断链表是否为空"""
        return self._head == None

    def create(self,data):
        self._head = SingleNode(0)
        cur = self._head
        for i in range(len(data)):
            node = SingleNode(data[i])
            cur.next = node
            cur = cur.next

    def length(self):
        """链表长度"""
        # cur初始时指向头节点
        cur = self._head
        count = 0
        while cur:
            count += 1
            # 将cur后移一个节点
            cur = cur.next
        return count

    def travel(self):
        """遍历链表"""
        cur = self._head
        while cur:
            print(cur.item)
            cur = cur.next

    def add(self, item):
        """头部添加元素"""
        # 先创建一个保存item值的节点
        node = SingleNode(item)
        # 将新节点的链接域next指向头节点，即_head指向的位置
        node.next = self._head
        # 将链表的头_head指向新节点
        self._head = node

    def append(self, item):
        """尾部添加元素"""
        node = SingleNode(item)
        # 先判断链表是否为空，若是空链表，则将_head指向新节点
        if self.is_empty():
            self._head = node
        # 若不为空，则找到尾部，将尾节点的next指向新节点
        else:
            cur = self._head
            while cur.next != None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        """指定位置添加元素"""
        # 若指定位置pos为第一个元素之前，则执行头部插入
        if pos <= 0:
            self.add(item)
        # 若指定位置超过链表尾部，则执行尾部插入
        elif pos > (self.length() - 1):
            self.append(item)
        # 找到指定位置
        else:
            node = SingleNode(item)
            count = 0
            # pre用来指向指定位置pos的前一个位置pos-1，初始从头节点开始移动到指定位置
            pre = self._head
            while count < (pos - 1):
                count += 1
                pre = pre.next
            # 先将新节点node的next指向插入位置的节点
            node.next = pre.next
            # 将插入位置的前一个节点的next指向新节点
            pre.next = node

    def remove(self, item):
        """删除节点"""
        cur = self._head
        pre = None
        while cur != None:
            # 找到了指定元素
            if cur.item == item:
                # 如果第一个就是删除的节点
                if not pre:
                    # 将头指针指向头节点的后一个节点
                    self._head = cur.next
                else:
                    # 将删除位置前一个节点的next指向删除位置的后一个节点
                    pre.next = cur.next
                break
            else:
                # 继续按链表后移节点
                pre = cur
                cur = cur.next

    def search(self, item):
        """链表查找节点是否存在，并返回True或者False"""
        cur = self._head
        while cur != None:
            if cur.item == item:
                return True
            cur = cur.next
        return False




```


```python
# 递归
def unfold_dict(v,host_name):
    for k,v in v.items():
        if isinstance(v,dict):
            host_name = k + '__'
            print('ok')
            unfold_dict(v,host_name=host_name)
        elif isinstance(v,str):
            try:
                json.loads(v)
            except:
                print('ok')
                xx[host_name+k] = v
        else:
            raise "error"
            
```

