"""Python Generic Abstract Base Classes Examples"""

from typing import Generic, TypeVar
from abc import ABC, abstractmethod

# 1. 泛型（Generic）
# 泛型是一种允许类或函数在定义时指定类型参数的机制。通过泛型，
# 你可以创建通用的代码结构，这些结构可以在不同的数据类型上工作，同时保持类型安全。
T = TypeVar('T')  # 定义一个类型变量

class Box(Generic[T]):
    """
    Box是一个简单的泛型类示例，它可以接受任何类型T的内容。
    你可以创建Box[int]、Box[str]等不同类型的实例。
    """
    def __init__(self, content: T):
        self.content = content

    def get_content(self) -> T:
        return self.content


# 2. 抽象基类（ABC）
# 抽象基类是一种不能被实例化的类，它用于定义接口或抽象方法，
# 这些方法必须在子类中实现。抽象基类通过abc模块中的ABC和abstractmethod装饰器来定义。
class Shape(ABC):
    """
    Shape是一个抽象基类示例，它定义了两个抽象方法area和perimeter。
    任何继承Shape的子类都必须实现这两个方法。
    """
    @abstractmethod
    def area(self) -> float:
        pass

    @abstractmethod
    def perimeter(self) -> float:
        pass


# 3. 泛型抽象基类
# 泛型抽象基类结合了泛型和抽象基类的特性。它是一个泛型类，同时也是一个抽象基类，
# 可以定义泛型的抽象方法，这些方法需要在子类中实现。
T = TypeVar('T')

class Stack(Generic[T], ABC):
    """
    Stack是一个泛型抽象基类示例，它定义了三个抽象方法push、pop和is_empty。
    任何继承Stack的子类都必须实现这些方法，并且可以指定具体的类型参数。
    """
    @abstractmethod
    def push(self, item: T) -> None:
        pass

    @abstractmethod
    def pop(self) -> T:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass


# 4. 实现泛型抽象基类
class ListStack(Stack[int]):
    """
    ListStack是Stack泛型抽象基类的一个实现，它处理整数类型的数据。
    ListStack实现了所有抽象方法，并且可以被实例化和使用。
    """
    def __init__(self):
        self.items: list[int] = []

    def push(self, item: int) -> None:
        self.items.append(item)

    def pop(self) -> int:
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Pop from empty stack")

    def is_empty(self) -> bool:
        return len(self.items) == 0


# 使用示例
if __name__ == "__main__":
    # Box泛型类示例
    int_box = Box[int](42)
    str_box = Box[str]("Hello Generic")
    print(f"Int box contains: {int_box.get_content()}")
    print(f"String box contains: {str_box.get_content()}")
    
    # Stack泛型抽象基类实现示例
    stack = ListStack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    
    print(f"Popped: {stack.pop()}")  # 输出 3
    print(f"Popped: {stack.pop()}")  # 输出 2
    print(f"Stack is empty: {stack.is_empty()}")  # 输出 False

# 总结：泛型抽象基类是一种结合了泛型和抽象基类特性的类。它允许你定义一个通用的接口或抽象方法，
# 这些方法可以处理多种数据类型，并且必须在子类中实现。这种设计在需要创建灵活且可扩展的代码结构时非常有用。
