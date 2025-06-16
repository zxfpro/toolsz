

## 桥接模式
class DesignerNotebook():
    def __init__(self):
        self.p = ''

    def __repr__(self):
        return self.p

class Connect(DesignerNotebook):
    """
使用效果就是直接调用老接口的对象的话对方咩有这个方法,
那么就将老接口传入一个适配器,这样,老接口就有了新方法 有点像电源适配器,或者插头转换器

    """
    def __init__(self,decorator):
        self.p = f"""

from abc import ABC, abstractmethod

# Implementor 接口
class Color(ABC):
    @abstractmethod
    def fill(self):
        pass

# Concrete Implementor A
class RedColor(Color):
    def fill(self):
        return "Filling with red color"

# Concrete Implementor B
class GreenColor(Color):
    def fill(self):
        return "Filling with green color"

# Abstraction 类
class Shape(ABC):
    def __init__(self, color: Color):
        self.color = color

    @abstractmethod
    def draw(self):
        pass

# Refined Abstraction A
class Circle(Shape):
    def draw(self):
        return f"Circle drawn. {self.color.fill()}"

# Refined Abstraction B
class Square(Shape):
    def draw(self):
        return f"Square drawn. {self.color.fill()}"

# 使用桥接模式
red = RedColor()
green = GreenColor()

circle = Circle(red)
square = Square(green)

print(circle.draw())
print(square.draw())
# 桥接模式通过组合关系（而非继承关系）来连接抽象和实现，从而实现更灵活的代码结构。

# 就要像你绘制好一幅地图,然后一个一个往前搭建
# 就好像枪械与配件一样, 我们定义枪的抽象,但是告诉你枪旁边的配件曹的尺寸,这样,你的配件就可以适应所有类型的枪,而不用管枪是谁生产的,同时,一把枪也可以适配大量的配件,而不用考虑配件是谁产的,什么性能
# 类似于现在的type-c 协议 http协议一样

"""

