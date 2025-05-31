# Mock 行为分析服务 (对应模块3)

# Depends on MockTransactionDataStorage and MockCodeInterpreterDataSource
from data.storage import MockTransactionDataStorage, MockCodeInterpreterDataSource

class AnalysisService:
    def __init__(self, transaction_storage: MockTransactionDataStorage, data_source: MockCodeInterpreterDataSource):
        self._transaction_storage = transaction_storage
        self._data_source = data_source # Use the data source for calculations
        print("AnalysisService initialized.")

    def perform_general_analysis(self):
        print("AnalysisService: Performing general spending analysis (mock)...")
        transactions = self._data_source.execute_query("获取所有交易") # Get data via data source
        if not transactions:
            print("AnalysisService: No transactions available for analysis.")
            return "无数据"

        total_expense = self._data_source.execute_query("计算总支出")

        # Simulate other analysis steps
        category_spending = {"餐饮": 150.0, "交通": 30.0, "健身": 100.0, "其他": 50.0} # Mock data
        highest_category = max(category_spending, key=category_spending.get) if category_spending else "N/A"
        spending_trend = "上升" # Mock trend

        analysis_result = {
            "total_expense": total_expense,
            "category_spending": category_spending,
            "highest_category": highest_category,
            "spending_trend": spending_trend
        }
        print("AnalysisService: General analysis complete (mock).")
        return analysis_result

    def perform_diet_analysis(self):
        print("AnalysisService: Performing diet analysis (mock)...")
        # Simulate retrieving and analyzing diet-related transactions
        diet_transactions = self._transaction_storage.get_transactions_by_category("餐饮") # Use mock storage getter
        if not diet_transactions:
             print("AnalysisService: No diet transactions available.")
             return "无数据"

        # Simulate identifying meal times, food types, etc.
        mock_meal_counts = {"早餐": 1, "午餐": 2, "晚餐": 3, "下午茶": 1, "外卖": 2}
        mock_food_types_spent = {"咖啡": 35.0, "快餐": 45.0, "正餐": 100.0} # Mock data
        mock_nutrition_estimate = {"calories_avg_daily": "未知 (需要更多数据)"} # Mock advanced feature

        analysis_result = {
            "meal_counts": mock_meal_counts,
            "food_types_spent": mock_food_types_spent,
            "nutrition_estimate": mock_nutrition_estimate,
            "raw_diet_transactions_count": len(diet_transactions)
        }
        print("AnalysisService: Diet analysis complete (mock).")
        return analysis_result

    def perform_exercise_analysis(self):
        print("AnalysisService: Performing exercise analysis (mock)...")
        # Simulate retrieving and analyzing exercise-related transactions
        exercise_transactions = self._transaction_storage.get_transactions_by_category("健身") # Use mock storage getter
        if not exercise_transactions:
             print("AnalysisService: No exercise transactions available.")
             return "无数据"

        mock_total_exercise_cost = sum(t.amount for t in exercise_transactions) # Simple mock sum
        mock_exercise_frequency = "未知 (需要更多数据)" # Mock pattern recognition

        analysis_result = {
            "total_exercise_cost": mock_total_exercise_cost,
            "exercise_frequency": mock_exercise_frequency,
            "raw_exercise_transactions_count": len(exercise_transactions)
        }
        print("AnalysisService: Exercise analysis complete (mock).")
        return analysis_result

    def perform_balance_analysis(self):
        print("AnalysisService: Performing diet-exercise balance analysis (mock)...")
        # Get results from other analyses (simulate calling other methods or getting pre-calculated data)
        diet_result = self.perform_diet_analysis()
        exercise_result = self.perform_exercise_analysis()

        if diet_result == "无数据" and exercise_result == "无数据":
            print("AnalysisService: Not enough data for balance analysis.")
            return "无数据"

        mock_diet_cost = sum(diet_result.get("food_types_spent", {}).values()) if diet_result != "无数据" else 0
        mock_exercise_cost = exercise_result.get("total_exercise_cost", 0) if exercise_result != "无数据" else 0

        mock_balance_ratio = f"{mock_diet_cost}:{mock_exercise_cost}" if mock_exercise_cost > 0 else f"{mock_diet_cost}:N/A"
        mock_qualitative_assessment = "初步看起来饮食支出高于运动支出。" # Mock qualitative assessment

        analysis_result = {
            "diet_cost_mock": mock_diet_cost,
            "exercise_cost_mock": mock_exercise_cost,
            "balance_ratio_mock": mock_balance_ratio,
            "assessment_mock": mock_qualitative_assessment
        }
        print("AnalysisService: Balance analysis complete (mock).")
        return analysis_result
