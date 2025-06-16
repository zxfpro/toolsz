
### 享元模式

class DesignerNotebook():
    def __init__(self):
        self.p = ''

    def __repr__(self):
        return self.p

class ShareYuan(DesignerNotebook):
    """
享元模式（Flyweight Pattern）是一种结构型设计模式，它通过共享细粒度对象来减少内存消耗并提高性能。享元模式特别适用于大量相似对象的场景，它通过共享相同的对象来减少重复实例的数量，从而节省内存。

    """
    def __init__(self,decorator):
        self.p = f"""
class TreeType:
    def __init__(self, name, color, texture):
        self.name = name
        self.color = color
        self.texture = texture

    def draw(self, canvas, x, y):
        # 在给定位置绘制树
        print(f"Drawing tree '{self.name}' of color '{self.color}' with texture '{self.texture}' at ({x}, {y})")

class TreeFactory:
    _tree_types = {123}

    @classmethod
    def get_tree_type(cls, name, color, texture):
        key = (name, color, texture)
        if key not in cls._tree_types:
            cls._tree_types[key] = TreeType(name, color, texture)
        return cls._tree_types[key]

class Tree:
    def __init__(self, x, y, tree_type):
        self.x = x
        self.y = y
        self.tree_type = tree_type

    def draw(self, canvas):
        self.tree_type.draw(canvas, self.x, self.y)

class Forest:
    def __init__(self):
        self.trees = []

    def plant_tree(self, x, y, name, color, texture):
        tree_type = TreeFactory.get_tree_type(name, color, texture)
        tree = Tree(x, y, tree_type)
        self.trees.append(tree)

    def draw(self, canvas):
        for tree in self.trees:
            tree.draw(canvas)

# 使用享元模式
forest = Forest()
forest.plant_tree(1, 1, "Oak", "Green", "Rough")
forest.plant_tree(2, 3, "Pine", "Green", "Smooth")
forest.plant_tree(4, 5, "Oak", "Green", "Rough")  # 共享已有的树种类

forest.draw("Canvas")

# 单例模式的加强版
#共享一部分内容 而平移另一部分内容
# 有点像图章,制作好了一个图章以后, 就可以到处印了
# 如果想要新的图章,就定义新图章

"""
