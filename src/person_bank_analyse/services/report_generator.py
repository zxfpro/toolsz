# Mock 报告生成服务 (对应模块4)



# Depends on AnalysisService and MockAIGuesser
class ReportGeneratorService:
    def __init__(self, analysis_service, ai_guesser):
        self._analysis_service = analysis_service
        self._ai_guesser = ai_guesser
        self._custom_report_items = [] # Mock storage for custom items
        print("ReportGeneratorService initialized.")

    def generate_fixed_monthly_report(self):
        print("ReportGeneratorService: Generating fixed monthly report (mock)...")
        # Simulate getting analysis results
        general_analysis = self._analysis_service.perform_general_analysis()
        diet_analysis = self._analysis_service.perform_diet_analysis()
        exercise_analysis = self._analysis_service.perform_exercise_analysis()

        # Use AI Guesser to draft sections
        report_parts = []
        report_parts.append("--- 固定月度个人行为分析报告 ---")
        report_parts.append("\n**消费概览:**")
        report_parts.append(self._ai_guesser.draft_report_section(general_analysis, "spending_summary"))
        report_parts.append(f"(Mock Data: 总支出 {general_analysis.get('total_expense', 'N/A')} 元)")

        report_parts.append("\n**饮食分析:**")
        if diet_analysis != "无数据":
             report_parts.append(self._ai_guesser.draft_report_section(diet_analysis, "diet_insight"))
             report_parts.append(f"(Mock Data: 餐饮交易 {diet_analysis.get('raw_diet_transactions_count', 0)} 笔, 咖啡支出 {diet_analysis.get('food_types_spent', {}).get('咖啡', 0)} 元)")
        else:
             report_parts.append("无足够的饮食数据进行分析。")


        report_parts.append("\n**运动健身分析:**")
        if exercise_analysis != "无数据":
             report_parts.append(self._ai_guesser.draft_report_section(exercise_analysis, "exercise_insight"))
             report_parts.append(f"(Mock Data: 运动支出 {exercise_analysis.get('total_exercise_cost', 0)} 元)")
        else:
             report_parts.append("无足够的运动数据进行分析。")


        report_parts.append("\n**自定义分析项目:**")
        if self._custom_report_items:
             report_parts.extend([f"- {item} (Mock Data: {item} value)" for item in self._custom_report_items])
        else:
             report_parts.append("当前无自定义分析项目。")


        report_parts.append("\n--- 报告结束 ---")

        report_text = "\n".join(report_parts)
        print("ReportGeneratorService: Fixed monthly report generated (mock).")
        return report_text

    def add_custom_item_to_report(self, item_description):
        print(f"ReportGeneratorService: Adding custom report item: '{item_description}' (mock).")
        self._custom_report_items.append(item_description)
        print("ReportGeneratorService: Custom item added.")
