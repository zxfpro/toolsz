# 领域模型 (Transaction, Category, UserRegistryEntry等)


class Transaction:
    def __init__(self, date, description, amount, type, raw_data=None):
        self.date = date # str or datetime
        self.description = description # str (raw商户描述)
        self.amount = amount # float
        self.type = type # str ("income" or "expense")
        self.raw_data = raw_data # Optional: store original row/dict

    def __repr__(self):
        return f"Transaction(Date: {self.date}, Desc: '{self.description}', Amount: {self.amount:.2f}, Type: {self.type})"

class Category:
    def __init__(self, name, parent=None):
        self.name = name # str (e.g., "餐饮", "交通")
        self.parent = parent # Optional: Parent category for hierarchy

    def __repr__(self):
        return f"Category(Name: '{self.name}')"

class UserRegistryEntry:
    def __init__(self, merchant_keyword, category_name, specific_info=None):
        self.merchant_keyword = merchant_keyword # str (用户输入的商户关键词)
        self.category_name = category_name # str (关联的类别名)
        self.specific_info = specific_info # Optional: dict (e.g., {"餐次": "午餐", "食物": "咖啡"})

    def __repr__(self):
        return f"UserRegistryEntry(Keyword: '{self.merchant_keyword}', Category: '{self.category_name}', Info: {self.specific_info})"

# Add more models as needed based on plan details (e.g., AnalysisReport, Insight, Suggestion)
# For this framework, these basic models are sufficient.