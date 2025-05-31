# Mock 数据存储层 (用户注册表, 交易数据)
from domain.models import UserRegistryEntry
from domain.models import Transaction

class MockUserRegistryStorage:
    def __init__(self):
        self._registry = [] # In-memory mock storage
        print("MockUserRegistryStorage initialized.")

    def load(self):
        print("Mock: Loading user registry data from storage.")
        # Simulate loading some data
        self._registry = [
            UserRegistryEntry("星巴克", "餐饮", {"餐次": "下午茶", "食物": "咖啡"}),
            UserRegistryEntry("滴滴出行", "交通"),
        ]
        print(f"Mock: Loaded {len(self._registry)} registry entries.")
        return self._registry

    def save(self, registry_data):
        print(f"Mock: Saving {len(registry_data)} user registry entries to storage.")
        self._registry = registry_data
        print("Mock: Save complete.")

    def add_entry(self, entry: UserRegistryEntry):
        print(f"Mock: Adding registry entry for '{entry.merchant_keyword}'.")
        self._registry.append(entry)
        print("Mock: Entry added.")

    def find_entry(self, merchant_keyword):
        print(f"Mock: Searching registry for keyword '{merchant_keyword}'.")
        # Simulate search
        for entry in self._registry:
            if merchant_keyword in entry.merchant_keyword: # Simple mock match
                print(f"Mock: Found entry for '{merchant_keyword}'.")
                return entry
        print(f"Mock: No entry found for '{merchant_keyword}'.")
        return None

    def get_all_entries(self):
        print("Mock: Retrieving all registry entries.")
        return list(self._registry) # Return a copy


class MockTransactionDataStorage:
    def __init__(self):
        self._transactions = [] # In-memory mock storage
        print("MockTransactionDataStorage initialized.")

    def save_transactions(self, transactions):
        print(f"Mock: Saving {len(transactions)} transactions.")
        self._transactions = transactions
        print("Mock: Transactions saved.")

    def get_all_transactions(self):
        print("Mock: Retrieving all transactions.")
        return list(self._transactions) # Return a copy

    def get_transactions_by_category(self, category_name):
        print(f"Mock: Retrieving transactions for category '{category_name}'.")
        # Simulate filtering
        return [t for t in self._transactions if hasattr(t, 'category') and t.category.name == category_name] # Assumes classification adds 'category' attr

    def get_transactions_in_date_range(self, start_date, end_date):
        print(f"Mock: Retrieving transactions from {start_date} to {end_date}.")
        # Simulate date filtering
        return self._transactions # Return all for simplicity in mock


# Base classes for Strategy pattern demonstration (Mock implementations)
class MockDataImporter:
    def import_data(self, source):
        print(f"MockDataImporter: Importing data from source: {source}")
        # Simulate parsing and creating transaction objects
        mock_transactions = [
            Transaction("2023-10-01", "星巴克消费", 35.0, "expense"),
            Transaction("2023-10-02", "滴滴出行", 15.5, "expense"),
            Transaction("2023-10-03", "工资收入", 8000.0, "income"),
            Transaction("2023-10-04", "美团外卖", 45.0, "expense"),
            Transaction("2023-10-05", "超级猩猩健身房", 100.0, "expense"),
        ]
        print(f"MockDataImporter: Successfully imported {len(mock_transactions)} mock transactions.")
        return mock_transactions

class MockCategoryClassifier:
    def classify(self, transaction: Transaction, user_registry, ai_guesser):
        print(f"MockCategoryClassifier: Classifying transaction: '{transaction.description}'")
        # Simulate classification logic using user registry, rules, AI
        category_name = "未知"
        specific_info = None

        # 1. Mock User Registry Lookup
        registry_entry = user_registry.find_entry(transaction.description)
        if registry_entry:
            category_name = registry_entry.category_name
            specific_info = registry_entry.specific_info
            print(f"MockCategoryClassifier: Matched user registry entry: Category '{category_name}'")
        # 2. Mock Rules/Keyword Matching (Simple)
        elif "健身" in transaction.description or "运动" in transaction.description:
             category_name = "健身"
             print(f"MockCategoryClassifier: Matched rule/keyword: Category '{category_name}'")
        elif "工资" in transaction.description or "收入" in transaction.description:
             category_name = "收入"
             print(f"MockCategoryClassifier: Matched rule/keyword: Category '{category_name}'")
        # 3. Mock AI Guessing
        else:
            print(f"MockCategoryClassifier: No registry/rule match, consulting Mock AI Guesser...")
            category_name, specific_info = ai_guesser.guess_category(transaction.description)
            print(f"MockCategoryClassifier: Mock AI guessed: Category '{category_name}'")

        # Attach classification result to the transaction object for downstream services
        setattr(transaction, 'category', Category(category_name))
        if specific_info:
             setattr(transaction, 'specific_info', specific_info)

        print(f"MockCategoryClassifier: Classified '{transaction.description}' as '{category_name}'.")
        return transaction

# Simple Mock AI component
class MockAIGuesser:
    def guess_category(self, description):
        print(f"MockAIGuesser: Simulating AI guess for '{description}'")
        # Very basic mock guessing
        if "咖啡" in description or "星巴克" in description or "瑞幸" in description:
            return "餐饮", {"食物": "咖啡"}
        elif "外卖" in description or "餐饮" in description:
             return "餐饮", {"餐次": "未知"}
        elif "出行" in description or "打车" in description:
             return "交通", None
        elif "购物" in description or "电商" in description:
             return "购物", None
        else:
             return "其他", None

    def draft_report_section(self, analysis_result, section_type):
         print(f"MockAIGuesser: Simulating AI drafting report section for '{section_type}'")
         # Mock AI response based on analysis type
         if section_type == "spending_summary":
              return "根据您的消费数据，本月总支出 mock_total_expense 元，其中餐饮占比 mock_food_percentage%。请注意mock_highest_category类目支出较高。"
         elif section_type == "diet_insight":
              return "饮食方面，本月有 mock_외식_count 次外出用餐/外卖。咖啡支出 mock_coffee_expense 元。请留意mock_diet_pattern等饮食模式。"
         elif section_type == "exercise_insight":
              return "健身方面，本月在运动健康上的支出为 mock_exercise_expense 元。请根据此信息规划您的运动安排。"
         elif section_type == "balance_insight":
              return "初步分析显示，您的饮食和运动支出比例为 mock_ratio。保持均衡是健康的关键。"
         else:
             return "MockAIGuesser: 无法生成该类型报告片段。"

    def answer_query(self, query, data_source):
         print(f"MockAIGuesser: Simulating AI understanding and answering query: '{query}'")
         # Mock AI response based on query keyword
         if "支出" in query or "花销" in query:
              print("MockAIGuesser: Query related to spending detected, instructing Code Interpreter (simulated)...")
              # In a real scenario, AI would parse query and tell Code Interpreter what data/calculation is needed
              mock_calculation_result = data_source.execute_query("计算总支出") # Simulate Code Interpreter call
              return f"好的，根据我的模拟计算，您的总支出是 {mock_calculation_result} 元。"
         elif "分类" in query or "类别" in query:
             print("MockAIGuesser: Query related to categories detected...")
             return "您想了解哪个类别的支出情况？"
         elif "注册表" in query or "自定义" in query:
             print("MockAIGuesser: Query related to registry detected...")
             return "您想如何管理您的自定义注册表？我可以帮您添加或查看。"
         else:
             return "抱歉，我不太理解您的问题。您可以问我关于支出、分类、报告等方面的问题。"


class MockCodeInterpreterDataSource:
     """
     Simulates the data query/calculation part of Code Interpreter.
     In a real app, this would interact with actual data storage/processing library (like pandas).
     """
     def __init__(self, transactions):
          self._transactions = transactions # Hold reference to loaded data
          print("MockCodeInterpreterDataSource initialized with transaction data.")

     def execute_query(self, query_command):
          print(f"MockCodeInterpreterDataSource: Simulating executing query command: '{query_command}'")
          # Simulate different query executions
          if query_command == "计算总支出":
               total = sum(t.amount for t in self._transactions if t.type == "expense")
               print(f"MockCodeInterpreterDataSource: Calculated total expense: {total}")
               return total
          elif query_command == "获取所有交易":
               print(f"MockCodeInterpreterDataSource: Returning {len(self._transactions)} transactions.")
               return self._transactions
          # Add more mock query commands as needed
          else:
               print(f"MockCodeInterpreterDataSource: Unknown query command '{query_command}'.")
               return None

# Simple Observer pattern implementation for learning mechanism
class Observer:
    def update(self, subject, event, data=None):
        pass # Abstract update method

class Subject:
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    def notify(self, event, data=None):
        print(f"Subject: Notifying observers about event '{event}'...")
        for observer in self._observers:
            observer.update(self, event, data)
        print("Subject: Notification complete.")