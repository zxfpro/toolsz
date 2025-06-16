""" 设计模式 """

class DesignerNotebook():
    def __init__(self):
        self.p = ''

    def __repr__(self):
        return self.p


class 观察者模式():
    # 订阅者 
    def update(self,class_name):
        self.p = """
from abc import ABC, abstractmethod
from typing import List

# 主题接口
class Subject(ABC):
    @abstractmethod
    def attach(self, observer: 'Observer'):
        pass

    @abstractmethod
    def detach(self, observer: 'Observer'):
        pass

    @abstractmethod
    def notify(self):
        pass

# 具体主题
class WeatherData(Subject):
    '''
    weather_data = WeatherData()
    print('################')
    temp_display1 = TemperatureDisplay("Main")
    temp_display2 = TemperatureDisplay("Secondary")
    print('################')
    weather_data.attach(temp_display1)
    weather_data.attach(temp_display2)
    print('################')
    weather_data.set_temperature(25.0)
    weather_data.set_temperature(30.0)
    print('################')
    weather_data.detach(temp_display1)
    weather_data.set_temperature(35.0)
    '''
    def __init__(self):
        self._observers: List[Observer] = []
        self._temperature: float = 0.0

    def attach(self, observer: 'Observer'):# 加载订阅者
        self._observers.append(observer)

    def detach(self, observer: 'Observer'):# 卸载订阅者
        self._observers.remove(observer)

    def notify(self):
        for observer in self._observers:
            observer.update(self._temperature)

    def set_temperature(self, temperature: float):
        print(f"WeatherData: setting temperature to {temperature}")
        self._temperature = temperature
        self.notify()

# 观察者接口
class Observer(ABC):
    @abstractmethod
    def update(self, temperature: float):
        pass

# 具体观察者
class TemperatureDisplay(Observer): #订阅者
    def __init__(self, name: str):
        self._name = name
        self._temperature = 0.0

    def update(self, temperature: float):
        self._temperature = temperature
        print(f"{self._name} Display: temperature updated to {self._temperature}")

"""
        print(self.p)
        return self.p
    






class 访问者模式():
    # 订阅者 
    def update(self,class_name):
        self.p = """
import math
# 我们来写一个简单的访问者模式 demo。这个 demo 将模拟一个图形结构，包含圆形和方形，并且我们将使用访问者模式来计算它们的面积和周长。
# 场景： 有一个图形的集合，包含圆形和方形。我们需要计算这个集合中所有图形的总面积和总周长。
# 使用访问者模式的优势： 如果将来需要添加新的图形类型（例如，三角形）或者新的计算方式（例如，计算重心），我们可以很容易地扩展，而无需修改原有的图形类。
# 1. Element 接口
class Element:
    def accept(self, visitor):
        pass

# 2. 具体 Element 类
class Circle(Element):
    def __init__(self, radius):
        self.radius = radius

    def accept(self, visitor):
        visitor.visit_circle(self)

    def get_radius(self):
        return self.radius

class Square(Element):
    def __init__(self, side):
        self.side = side

    def accept(self, visitor):
        visitor.visit_square(self)

    def get_side(self):
        return self.side

# 3. Visitor 接口
class Visitor:
    def visit_circle(self, circle):
        pass

    def visit_square(self, square):
        pass

# 4. 具体 Visitor 类
class AreaCalculatorVisitor(Visitor):
    def __init__(self):
        self._total_area = 0

    def visit_circle(self, circle):
        area = math.pi * circle.get_radius() ** 2
        print(f"Calculating area of Circle (radius={circle.get_radius()}): {area:.2f}")
        self._total_area += area

    def visit_square(self, square):
        area = square.get_side() ** 2
        print(f"Calculating area of Square (side={square.get_side()}): {area:.2f}")
        self._total_area += area

    def get_total_area(self):
        return self._total_area

class PerimeterCalculatorVisitor(Visitor):
    def __init__(self):
        self._total_perimeter = 0

    def visit_circle(self, circle):
        perimeter = 2 * math.pi * circle.get_radius()
        print(f"Calculating perimeter of Circle (radius={circle.get_radius()}): {perimeter:.2f}")
        self._total_perimeter += perimeter

    def visit_square(self, square):
        perimeter = 4 * square.get_side()
        print(f"Calculating perimeter of Square (side={square.get_side()}): {perimeter:.2f}")
        self._total_perimeter += perimeter

    def get_total_perimeter(self):
        return self._total_perimeter

# 5. 客户端代码
if __name__ == "__main__":
    # 创建图形对象
    shapes = [
        Circle(5),
        Square(4),
        Circle(3),
        Square(6)
    ]

    # 创建面积计算访问者
    area_calculator = AreaCalculatorVisitor()

    # 遍历图形，接受面积计算访问者
    print("--- Calculating Areas ---")
    for shape in shapes:
        shape.accept(area_calculator)

    print(f"\nTotal Area: {area_calculator.get_total_area():.2f}")

    # 创建周长计算访问者
    perimeter_calculator = PerimeterCalculatorVisitor()

    # 遍历图形，接受周长计算访问者
    print("\n--- Calculating Perimeters ---")
    for shape in shapes:
        shape.accept(perimeter_calculator)

    print(f"\nTotal Perimeter: {perimeter_calculator.get_total_perimeter():.2f}")

    # 假设将来需要添加新的操作，例如计算对角线长度 (对于方形)
    class DiagonalCalculatorVisitor(Visitor):
        def __init__(self):
            self._total_diagonal = 0

        def visit_circle(self, circle):
            # 圆形没有对角线，可以忽略或者打印提示
            print(f"Circle (radius={circle.get_radius()}) has no diagonal.")
            pass

        def visit_square(self, square):
            diagonal = math.sqrt(2) * square.get_side()
            print(f"Calculating diagonal of Square (side={square.get_side()}): {diagonal:.2f}")
            self._total_diagonal += diagonal

        def get_total_diagonal(self):
            return self._total_diagonal

    # 创建对角线计算访问者
    diagonal_calculator = DiagonalCalculatorVisitor()

    # 遍历图形，接受对角线计算访问者
    print("\n--- Calculating Diagonals ---")
    for shape in shapes:
        shape.accept(diagonal_calculator)

    print(f"\nTotal Diagonal (of squares): {diagonal_calculator.get_total_diagonal():.2f}")

"""
        print(self.p)
        return self.p
    




class 解释器模式():
    # 订阅者 
    """
    解释器模式的应用场景
编译器：编译器中的语法解析器使用解释器模式来解析和解释代码。
脚本语言：解释脚本语言，如正则表达式解析器、SQL解析器等。
配置文件解析：解析和评估简单的配置文件或命令行参数。
机器人指令解释：机器人或自动化系统中的命令解释和执行。

没太懂
    """
    def update(self,class_name):
        self.p = """
from abc import ABC, abstractmethod

# 上下文类，存储变量的值
class Context:
    def __init__(self):
        self.data = {}
    
    def set(self, variable, value):
        self.data[variable] = value
    
    def get(self, variable):
        return self.data.get(variable)

# 抽象表达式类
class Expression(ABC):
    @abstractmethod
    def interpret(self, context: Context):
        pass

# 终结符表达式：表示变量
class VariableExpression(Expression):
    def __init__(self, name):
        self.name = name
    
    def interpret(self, context: Context):
        return context.get(self.name)

# 终结符表达式：表示数字
class NumberExpression(Expression):
    def __init__(self, number):
        self.number = number
    
    def interpret(self, context: Context):
        return self.number

# 非终结符表达式：加法表达式
class AddExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
    
    def interpret(self, context: Context):
        return self.left.interpret(context) + self.right.interpret(context)

# 非终结符表达式：减法表达式
class SubtractExpression(Expression):
    def __init__(self, left: Expression, right: Expression):
        self.left = left
        self.right = right
    
    def interpret(self, context: Context):
        return self.left.interpret(context) - self.right.interpret(context)

# 客户端代码
if __name__ == "__main__":
    # 创建上下文并设置变量的值
    context = Context()
    context.set("x", 10)
    context.set("y", 20)

    # 创建表达式
    expression = AddExpression(
        SubtractExpression(
            NumberExpression(5),
            VariableExpression("x")
        ),
        VariableExpression("y")
    )

    # 解释并计算表达式的值
    result = expression.interpret(context)
    print(f"Result of the expression: {result}")  # 输出：Result of the expression: 15


"""

        print(self.p)
        return self.p
    