# 状态模式实现


import abc
# from cli.commands import CLICommand # Import commands
from domain.models import Transaction # Need model for ConfirmationState

class CLIState(abc.ABC):
    def __init__(self, app: 'CLIApp'):
        self._app = app # Reference to the main app/context

    @abc.abstractmethod
    def display_menu(self):
        """Displays the options for the current state."""
        pass

    @abc.abstractmethod
    def handle_input(self, user_input: str) -> 'CLIState':
        """Handles user input and returns the next state."""
        pass

class MainMenuState(CLIState):
    def display_menu(self):
        print("\n--- 主菜单 ---")
        print("请选择操作:")
        print("1. 导入数据")
        print("2. 分类交易")
        print("3. 进行分析")
        print("4. 生成报告")
        print("5. 自然语言查询")
        print("6. 查看注册表")
        print("0. 退出")

    def handle_input(self, user_input: str) -> CLIState:
        user_input = user_input.strip()
        if user_input == '1':
            return self._app.execute_command(ImportDataCommand(self._app.data_import_service, self._app.transaction_storage))
        elif user_input == '2':
            return self._app.execute_command(ClassifyCommand(self._app.classification_service, self._app.transaction_storage))
        elif user_input == '3':
            return self._app.execute_command(AnalyzeCommand(self._app.analysis_service))
        elif user_input == '4':
            return self._app.execute_command(ReportCommand(self._app.report_generator_service))
        elif user_input == '5':
             return self._app.execute_command(QueryCommand(self._app.interaction_service))
        elif user_input == '6':
             return self._app.execute_command(ShowRegistryCommand(self._app.user_registry_storage))
        elif user_input == '0':
            return self._app.execute_command(ExitCommand())
        else:
            print("无效输入，请重试。")
            return self # Stay in current state

class ConfirmationState(CLIState):
    def __init__(self, app: 'CLIApp', transactions_to_confirm):
        super().__init__(app)
        self._transactions_to_confirm = transactions_to_confirm
        self._current_index = 0
        print("\n--- 需要确认的交易 ---")
        print("AI 猜测或规则无法确定以下交易的准确分类，请您确认或提供更多信息。")

    def display_menu(self):
        if self._current_index < len(self._transactions_to_confirm):
            tx = self._transactions_to_confirm[self._current_index]
            print(f"\n当前交易 ({self._current_index + 1}/{len(self._transactions_to_confirm)}):")
            print(f"  描述: '{tx.description}'")
            print(f"  金额: {tx.amount:.2f}")
            print(f"  AI 初步猜测类别: {getattr(tx, 'category', 'N/A').name}") # Show AI's mock guess
            print("\n请提供正确的类别 (例如: 餐饮, 交通, 购物) 或输入 '跳过' / '退出':")
        else:
            print("\n所有需要确认的交易已处理。")
            from cli.states import MainMenuState # Import here
            self._app.set_data_classified(True) # Data is now classified (or partially classified)
            return MainMenuState(self._app)


    def handle_input(self, user_input: str) -> CLIState:
        user_input = user_input.strip()

        if self._current_index >= len(self._transactions_to_confirm):
             # Should not happen if display_menu is called first, but as a safeguard
             from cli.states import MainMenuState
             return MainMenuState(self._app)

        current_tx = self._transactions_to_confirm[self._current_index]

        if user_input.lower() == '退出':
             print("退出确认流程。")
             from cli.states import MainMenuState
             return MainMenuState(self._app) # Go back to main menu

        if user_input.lower() == '跳过':
             print(f"跳过交易 '{current_tx.description}' 的确认。")
             self._current_index += 1
             return self # Stay in ConfirmationState

        # Assume user input is the category name for now
        category_name = user_input
        specific_info_input = input(f"请输入关于 '{current_tx.description}' 的更详细信息 (例如: 餐次:午餐 食物:咖啡, 可选): ").strip()
        specific_info = {}
        if specific_info_input:
             # Simple mock parsing: expect "key:value key:value"
             try:
                 parts = specific_info_input.split()
                 for part in parts:
                     key_value = part.split(":")
                     if len(key_value) == 2:
                         specific_info[key_value[0].strip()] = key_value[1].strip()
                 print(f"Mock: 解析到详细信息: {specific_info}")
             except Exception as e:
                  print(f"Mock: 解析详细信息失败 ({e})。将忽略详细信息。")
                  specific_info = None


        # Call the service to confirm and update registry (mock)
        self._app.classification_service.confirm_classification(current_tx, category_name, specific_info if specific_info else None)

        print(f"Mock: 已确认交易 '{current_tx.description}' 为 '{category_name}'.")

        self._current_index += 1

        if self._current_index < len(self._transactions_to_confirm):
             return self # More transactions to confirm, stay in this state
        else:
             print("\n所有需要确认的交易已处理。")
             from cli.states import MainMenuState
             self._app.set_data_classified(True) # Data is now classified (or partially classified)
             return MainMenuState(self._app) # All done, go back to main menu