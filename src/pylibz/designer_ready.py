""" 设计模式 """

## 备忘录模式


```python

# @title 备忘录模式
#@markdown Please enter your details below:
# 主要用于实现状态回滚功能,回滚操作
class Memento:
    def __init__(self, state: str):
        self._state = state

    def get_state(self) -> str:
        return self._state

    def set_state(self, state: str):
        self._state = state

class Originator:
    def __init__(self):
        self._state = ""

    def set_state(self, state: str):
        self._state = state

    def get_state(self) -> str:
        return self._state

    def create_memento(self) -> Memento:
        return Memento(self._state)

    def set_memento(self, memento: Memento):
        self._state = memento.get_state()

class Caretaker:
    def __init__(self):
        self._memento = None

    def save(self, memento: Memento):
        self._memento = memento

    def retrieve(self) -> Memento:
        return self._memento

# 客户端代码
if __name__ == "__main__":
    originator = Originator()
    caretaker = Caretaker()

    # 设置和保存状态
    originator.set_state("State 1")
    print(f"Current State: {originator.get_state()}")
    caretaker.save(originator.create_memento())

    # 修改状态
    originator.set_state("State 2")
    print(f"Current State: {originator.get_state()}")

    # 恢复到先前的状态
    originator.set_memento(caretaker.retrieve())
    print(f"Restored State: {originator.get_state()}")








```python

## 命令模式

命令模式（Command Pattern）是一种行为型设计模式，它将请求封装为对象，使得可以使用不同的请求、队列或日志来参数化其他对象。命令模式还支持撤销操作。



from abc import ABC, abstractmethod

# 命令接口
class Command(ABC):
    @abstractmethod
    def execute(self):
        pass

# 接收者
class Light:
    def on(self):
        print("Light is ON")
    
    def off(self):
        print("Light is OFF")

# 具体命令 - 打开灯
class LightOnCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.on()

# 具体命令 - 关闭灯
class LightOffCommand(Command):
    def __init__(self, light: Light):
        self.light = light
    
    def execute(self):
        self.light.off()

# 调用者
class RemoteControl:
    def __init__(self):
        self.command = None
    
    def set_command(self, command: Command):
        self.command = command
    
    def press_button(self):
        if self.command:
            self.command.execute()

# 客户端代码
if __name__ == "__main__":
    # 创建接收者
    light = Light()

    # 创建具体命令
    light_on = LightOnCommand(light)
    light_off = LightOffCommand(light)

    # 创建调用者
    remote = RemoteControl()

    # 设置命令并执行
    remote.set_command(light_on)
    remote.press_button()  # 输出：Light is ON

    remote.set_command(light_off)
    remote.press_button()  # 输出：Light is OFF


命令模式的优点和缺点
优点：

解耦请求发送者和接收者：请求发送者只需知道命令对象，而不需要知道接收者是谁或如何处理请求。
支持撤销和恢复操作：通过存储命令对象，可以轻松实现撤销和恢复功能。
支持命令队列：命令对象可以排队执行，使得系统可以支持请求的日志记录和事务管理。
增加新的命令容易：添加新的命令只需实现命令接口，不需要修改现有的代码。
缺点：

增加了系统的复杂性：需要额外的类来实现命令模式，可能会增加系统的复杂性。
命令对象的管理：在大型系统中，管理大量的命令对象可能会变得困难。
命令模式的应用场景
GUI开发：如菜单项和按钮的操作，通过命令模式实现可以方便地实现撤销和恢复功能。
事务系统：如数据库事务，通过命令模式实现可以方便地支持事务的提交和回滚。
任务队列：如任务调度系统，通过命令模式可以将任务对象放入队列中，并按顺序执行。
日志和审计：通过命令模式，可以记录所有操作，以便于审计和回放。
通过命令模式，可以将请求的发送者和接收者解耦，使得系统更加灵活和可扩展。命令模式特别适用于需要支持撤销、恢复和事务管理的系统。

```











1. 依赖注入（Dependency Injection）
依赖注入是一种设计模式，通过将类的依赖项注入到类的实例中，而不是由类自行创建或管理其依赖项。这种模式促进了代码的松耦合和可测试性。

2. 生产者-消费者模式（Producer-Consumer Pattern）
前面已经介绍过，该模式解决了生产者和消费者之间的同步问题，常用于多线程环境下的数据共享和同步。

3. 发布-订阅模式（Publish-Subscribe Pattern）
发布-订阅模式允许发送者（发布者）和接收者（订阅者）之间的松耦合，发布者发布消息，订阅者订阅消息，消息通过消息通道传递。常用于事件驱动系统和消息队列系统。

4. 资源池模式（Object Pool Pattern）
资源池模式管理一个对象池，这些对象可以被重复使用，而不是每次都重新创建和销毁。这种模式适用于管理连接池、线程池、内存池等。

4. 黑板模式（Blackboard Pattern）
黑板模式用于解决复杂问题，涉及多个专家系统或知识源。所有组件都通过一个共享的黑板进行通信和协作。
"""










## 责任链



```python
from abc import ABC, abstractmethod

class Handler(ABC):
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler

    @abstractmethod
    def handle(self, request):
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class InfoHandler(Handler):
    def handle(self, request):
        if request == "INFO":
            print("InfoHandler: Handling INFO level request")
        else:
            super().handle(request)

class DebugHandler(Handler):
    def handle(self, request):
        if request == "DEBUG":
            print("DebugHandler: Handling DEBUG level request")
        else:
            super().handle(request)

class ErrorHandler(Handler):
    def handle(self, request):
        if request == "ERROR":
            print("ErrorHandler: Handling ERROR level request")
        else:
            super().handle(request)

# 客户端代码
if __name__ == "__main__":
    # 创建具体处理者
    info_handler = InfoHandler()
    debug_handler = DebugHandler()
    error_handler = ErrorHandler()

    # 设置处理链
    info_handler.set_next(debug_handler).set_next(error_handler)

    # 提交请求
    requests = ["INFO", "DEBUG", "ERROR", "UNKNOWN"]
    for req in requests:
        print(f"Client: Submitting {req} request")
        info_handler.handle(req)
        print()


```

OA 审批流程,权限验证





```python



## 中介者模式（Mediator）

简单来说就是群聊, 广播

from abc import ABC, abstractmethod

# 中介者接口
class Mediator(ABC):
    @abstractmethod
    def send(self, message: str, colleague: 'Colleague'):
        pass

# 具体中介者
class ConcreteMediator(Mediator):
    def __init__(self):
        self._colleagues = []

    def add_colleague(self, colleague: 'Colleague'):
        self._colleagues.append(colleague)
        colleague.set_mediator(self)

    def send(self, message: str, colleague: 'Colleague'):
        for c in self._colleagues:
            if c != colleague:
                c.receive(message)

# 同事对象接口
class Colleague(ABC):
    def __init__(self):
        self._mediator = None

    def set_mediator(self, mediator: Mediator):
        self._mediator = mediator

    @abstractmethod
    def send(self, message: str):
        pass

    @abstractmethod
    def receive(self, message: str):
        pass

# 具体同事对象
class ConcreteColleague(Colleague):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def send(self, message: str):
        print(f"{self._name} sends message: {message}")
        self._mediator.send(message, self)

    def receive(self, message: str):
        print(f"{self._name} receives message: {message}")

# 客户端代码
if __name__ == "__main__":
    # 创建中介者
    mediator = ConcreteMediator()

    # 创建同事对象
    colleague1 = ConcreteColleague("User1")
    colleague2 = ConcreteColleague("User2")
    colleague3 = ConcreteColleague("User3")

    # 将同事对象添加到中介者
    mediator.add_colleague(colleague1)
    mediator.add_colleague(colleague2)
    mediator.add_colleague(colleague3)

    # 发送消息
    colleague1.send("Hello everyone!")


```



```python





#@title 状态模式（State）
#状态模式允许对象在其内部状态发生改变时改变其行为，对象看起来好像修改了它的类。状态模式将状态转换行为封装到不同的状态类中，使得状态的切换更加灵活和可扩展。
#应用场景：状态机、游戏中的对象状态变化、工作流引擎等。
#开关灯的场景状态等,米家工作流.

from abc import ABC, abstractmethod

# 抽象状态
class State(ABC):
    @abstractmethod
    def handle(self, context: 'Context'):
        pass

# 具体状态：灯打开
class OnState(State):
    def handle(self, context: 'Context'):
        print("Light is already on. Turning it off.")
        context.set_state(OffState())

# 具体状态：灯关闭
class OffState(State):
    def handle(self, context: 'Context'):
        print("Light is off. Turning it on.")
        context.set_state(OnState())

# 上下文
class Context:
    def __init__(self, state: State):
        self._state = state

    def request(self):
        self._state.handle(self)

    def set_state(self, state: State):
        self._state = state

# 客户端代码
if __name__ == "__main__":
    # 创建上下文和初始状态
    context = Context(OffState())

    # 改变状态
    context.request()
    context.request()
    context.request()




# @title 策略模式（Strategy）
策略模式定义一系列算法，并将每个算法封装起来，使它们可以相互替换。策略模式使得算法可以在不影响客户端的情况下发生变化。

应用场景：排序算法、加密算法、日志策略等。
策略模式做的情况是,可以在不需要关闭服务的情况下,动态的变换策略
和工厂模式有点像

from abc import ABC, abstractmethod
from typing import List

# 策略接口
class SortStrategy(ABC):
    @abstractmethod
    def sort(self, data: List[int]) -> List[int]:
        pass

# 具体策略：快速排序
class QuickSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("Using QuickSort")
        return sorted(data)  # 这里使用Python内置排序作为简化的快速排序实现

# 具体策略：插入排序
class InsertionSortStrategy(SortStrategy):
    def sort(self, data: List[int]) -> List[int]:
        print("Using InsertionSort")
        for i in range(1, len(data)):
            key = data[i]
            j = i - 1
            while j >= 0 and key < data[j]:
                data[j + 1] = data[j]
                j -= 1
            data[j + 1] = key
        return data

# 上下文
class SortContext:
    def __init__(self, strategy: SortStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: SortStrategy):
        self._strategy = strategy

    def sort(self, data: List[int]) -> List[int]:
        return self._strategy.sort(data)

# 客户端代码
if __name__ == "__main__":
    data = [5, 3, 8, 4, 2]

    context = SortContext(QuickSortStrategy())
    print(context.sort(data))

    context.set_strategy(InsertionSortStrategy())
    print(context.sort(data))





