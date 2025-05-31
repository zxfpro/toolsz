# Mock 洞察与建议服务 (对应模块5)
# Depends on AnalysisService and MockAIGuesser
class InsightService:
    def __init__(self, analysis_service, ai_guesser):
        self._analysis_service = analysis_service
        self._ai_guesser = ai_guesser
        print("InsightService initialized.")

    def get_insights_and_suggestions(self):
        print("InsightService: Generating insights and suggestions (mock)...")
        # Get analysis results
        general_analysis = self._analysis_service.perform_general_analysis()
        balance_analysis = self._analysis_service.perform_balance_analysis()

        insights = []
        suggestions = []

        # Simulate generating insights based on mock analysis results
        if general_analysis != "无数据":
             insights.append(f"Mock Insight: 您的最高消费类别是 {general_analysis.get('highest_category', 'N/A')}。")
             if general_analysis.get("spending_trend") == "上升":
                  suggestions.append("Mock Suggestion: 您的消费趋势似乎正在上升，建议关注支出构成。")

        if balance_analysis != "无数据":
             insights.append(f"Mock Insight: 饮食与运动支出的模拟比例是 {balance_analysis.get('balance_ratio_mock', 'N/A')}。")
             suggestions.append(f"Mock Suggestion: 保持饮食与运动的均衡至关重要，{balance_analysis.get('assessment_mock', '')}")


        if not insights and not suggestions:
             insights.append("Mock Insight: 当前无足够的分析数据生成洞察。")
             suggestions.append("Mock Suggestion: 导入更多数据以获得个性化建议。")


        print("InsightService: Insights and suggestions generated (mock).")
        return {"insights": insights, "suggestions": suggestions}

