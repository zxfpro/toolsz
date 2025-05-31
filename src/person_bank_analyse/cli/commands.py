# 命令模式实现


import abc
from services.data_import import DataImportService
from services.classification import ClassificationService
from services.analysis import AnalysisService
from services.report_generator import ReportGeneratorService
from services.insight_service import InsightService
from services.interaction import InteractionService
from data.storage import MockUserRegistryStorage, MockTransactionDataStorage # Need storage for some commands
from domain.models import UserRegistryEntry, Transaction # Need models

# Forward declaration for type hinting circular dependency
from cli.states import CLIState

class CLICommand(abc.ABC):
    @abc.abstractmethod
    def execute(self, app: 'CLIApp') -> 'CLIState':
        """Executes the command and returns the next state."""
        pass

class ImportDataCommand(CLICommand):
    def __init__(self, import_service: DataImportService, transaction_storage: MockTransactionDataStorage):
        self._import_service = import_service
        self._transaction_storage = transaction_storage

    def execute(self, app: 'CLIApp') -> 'CLIState':
        print("\n--- 数据导入 ---")
        source_path = input("请输入数据源路径 (mock, e.g., transactions.csv): ")
        imported_count = self._import_service.import_from_source(source_path)
        print(f"Mock: 成功导入 {imported_count} 条交易记录。")
        # After import, data is available for classification and analysis
        app.set_data_loaded(True)
        from cli.states import MainMenuState # Import here to avoid circular dep
        return MainMenuState(app)

class ClassifyCommand(CLICommand):
    def __init__(self, classification_service: ClassificationService, transaction_storage: MockTransactionDataStorage):
        self._classification_service = classification_service
        self._transaction_storage = transaction_storage

    def execute(self, app: 'CLIApp') -> 'CLIState':
        print("\n--- 商户识别与分类 ---")
        transactions = self._transaction_storage.get_all_transactions()
        if not transactions:
            print("无交易数据可分类。请先导入数据。")
            from cli.states import MainMenuState
            return MainMenuState(app)

        print(f"Mock: 对 {len(transactions)} 条交易进行分类...")
        classified_transactions = self._classification_service.classify_transactions(transactions)

        transactions_to_confirm = self._classification_service.get_transactions_needing_confirmation()
        if transactions_to_confirm:
             print(f"\n有 {len(transactions_to_confirm)} 条交易需要用户确认。")
             from cli.states import ConfirmationState # Import here
             return ConfirmationState(app, transactions_to_confirm)
        else:
             print("\nMock: 所有交易分类完成，无需确认。")
             app.set_data_classified(True) # Mark data as classified
             from cli.states import MainMenuState
             return MainMenuState(app)


class AnalyzeCommand(CLICommand):
    def __init__(self, analysis_service: AnalysisService):
        self._analysis_service = analysis_service

    def execute(self, app: 'CLIApp') -> 'CLIState':
        print("\n--- 行为模式识别与精细化分析 ---")
        if not app.is_data_classified():
            print("请先完成数据导入和分类。")
            from cli.states import MainMenuState
            return MainMenuState(app)

        print("请选择分析类型:")
        print("1. 通用消费分析")
        print("2. 精细化饮食分析")
        print("3. 运动健身分析")
        print("4. 饮食与运动平衡分析")
        print("b. 返回主菜单")

        while True:
            choice = input("选择分析类型 (1-4, b): ").lower().strip()
            if choice == 'b':
                from cli.states import MainMenuState
                return MainMenuState(app)
            elif choice == '1':
                result = self._analysis_service.perform_general_analysis()
                print("\n--- 通用消费分析结果 (Mock) ---")
                print(result)
                break
            elif choice == '2':
                result = self._analysis_service.perform_diet_analysis()
                print("\n--- 精细化饮食分析结果 (Mock) ---")
                print(result)
                break
            elif choice == '3':
                result = self._analysis_service.perform_exercise_analysis()
                print("\n--- 运动健身分析结果 (Mock) ---")
                print(result)
                break
            elif choice == '4':
                result = self._analysis_service.perform_balance_analysis()
                print("\n--- 饮食与运动平衡分析结果 (Mock) ---")
                print(result)
                break
            else:
                print("无效选择。")

        print("\n分析完成。")
        from cli.states import MainMenuState
        return MainMenuState(app) # Always return to main menu after analysis

class ReportCommand(CLICommand):
    def __init__(self, report_service: ReportGeneratorService):
        self._report_service = report_service

    def execute(self, app: 'CLIApp') -> 'CLIState':
        print("\n--- 报告生成 ---")
        if not app.is_data_classified(): # Analysis relies on classified data
             print("请先完成数据导入和分类，并进行分析。")
             from cli.states import MainMenuState
             return MainMenuState(app)


        print("请选择报告类型:")
        print("1. 固定月度报告")
        print("2. 添加自定义报告项")
        print("b. 返回主菜单")

        while True:
            choice = input("选择报告类型 (1-2, b): ").lower().strip()
            if choice == 'b':
                from cli.states import MainMenuState
                return MainMenuState(app)
            elif choice == '1':
                report_text = self._report_service.generate_fixed_monthly_report()
                print("\n--- 生成报告 (Mock) ---")
                print(report_text)
                break
            elif choice == '2':
                item = input("请输入要添加到报告的自定义项目描述 (mock): ")
                self._report_service.add_custom_item_to_report(item)
                print(f"Mock: 已将 '{item}' 添加到自定义报告项列表。")
                # After adding, can choose to generate the report or add more
                continue # Stay in report options
            else:
                print("无效选择。")

        print("\n报告处理完成。")
        from cli.states import MainMenuState
        return MainMenuState(app)

class QueryCommand(CLICommand):
     def __init__(self, interaction_service: InteractionService):
          self._interaction_service = interaction_service

     def execute(self, app: 'CLIApp') -> 'CLIState':
          print("\n--- 自然语言交互查询 ---")
          if not app.is_data_classified():
               print("请先完成数据导入和分类。")
               from cli.states import MainMenuState
               return MainMenuState(app)

          print("请输入您的查询 (输入 '退出' 返回):")
          while True:
               query = input("> ").strip()
               if query.lower() == '退出':
                    break

               if not query:
                   continue

               response = self._interaction_service.handle_natural_language_query(query)
               print(f"AI Response (Mock): {response}")

          print("退出自然语言查询。")
          from cli.states import MainMenuState
          return MainMenuState(app)


class ShowRegistryCommand(CLICommand):
     def __init__(self, user_registry_storage: MockUserRegistryStorage):
          self._user_registry_storage = user_registry_storage

     def execute(self, app: 'CLIApp') -> 'CLIState':
          print("\n--- 用户自定义注册表 ---")
          registry_entries = self._user_registry_storage.get_all_entries()
          if not registry_entries:
               print("注册表当前为空。")
          else:
               print("以下是您的自定义注册表条目:")
               for entry in registry_entries:
                    print(f"- {entry}")

          print("\n注册表查看完成。")
          from cli.states import MainMenuState
          return MainMenuState(app)


class ExitCommand(CLICommand):
    def execute(self, app: 'CLIApp') -> 'CLIState':
        print("\n退出程序。感谢使用！")
        app.stop() # Tell the app to stop the loop
        return None # No next state