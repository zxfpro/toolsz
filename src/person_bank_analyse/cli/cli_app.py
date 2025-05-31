# CLI 应用主循环和状态机


from cli.states import CLIState, MainMenuState
from services.data_import import DataImportService
from services.classification import ClassificationService
from services.analysis import AnalysisService
from services.report_generator import ReportGeneratorService
from services.insight_service import InsightService
from services.interaction import InteractionService
from data.storage import MockUserRegistryStorage, MockTransactionDataStorage, MockDataImporter, MockCategoryClassifier, MockAIGuesser, MockCodeInterpreterDataSource

class CLIApp:
    def __init__(self):
        # Initialize Mock Data Storage and AI components
        self.user_registry_storage = MockUserRegistryStorage()
        self.transaction_storage = MockTransactionDataStorage()
        self.ai_guesser = MockAIGuesser()
        self.code_interpreter_data_source = MockCodeInterpreterDataSource(self.transaction_storage.get_all_transactions()) # Pass transaction data ref

        # Initialize Mock Strategies/Components
        self.data_importer = MockDataImporter()
        self.category_classifier = MockCategoryClassifier()

        # Initialize Mock Services (Dependency Injection)
        self.data_import_service = DataImportService(self.data_importer, self.transaction_storage)
        self.classification_service = ClassificationService(self.category_classifier, self.user_registry_storage, self.ai_guesser)
        self.analysis_service = AnalysisService(self.transaction_storage, self.code_interpreter_data_source) # Pass both storage and data_source
        self.report_generator_service = ReportGeneratorService(self.analysis_service, self.ai_guesser)
        self.insight_service = InsightService(self.analysis_service, self.ai_guesser)
        self.interaction_service = InteractionService(self.ai_guesser, self.code_interpreter_data_source)

        # Attach observers (Example: ConfirmationState observing ClassificationService)
        # This setup needs to be dynamic or managed carefully.
        # For simplicity in this framework, the ConfirmationState constructor implicitly links
        # or we could pass the app instance to services to allow them to access/notify states.
        # A better DI approach or a central event bus would be used in a real app.
        # For now, let's assume states can access the app which holds service refs.
        # The ClassificationService notifies generic "needs_confirmation" event;
        # The CLIApp/State handles transitioning *to* ConfirmationState based on command result.
        # However, the ClassificationService *confirm_classification* method *also* notifies,
        # which is where an observer (like the registry storage) could be attached.
        # Let's attach the registry storage as an observer for confirmation events.
        # This makes the registry storage update when ClassificationService notifies after confirmation.
        self.classification_service.attach(self.user_registry_storage) # MockUserRegistryStorage update() will be called

        self._running = False
        self._current_state: CLIState = MainMenuState(self)

        # Flags to track data state
        self._data_loaded = False
        self._data_classified = False

        print("\n--- 个人行为分析程序 (Mock Framework) ---")
        print("欢迎使用！这是一个演示程序框架，所有核心功能均为模拟实现。")


    def set_data_loaded(self, status: bool):
         self._data_loaded = status
         # Update the data source reference after loading new data
         self.code_interpreter_data_source = MockCodeInterpreterDataSource(self.transaction_storage.get_all_transactions())
         self.analysis_service = AnalysisService(self.transaction_storage, self.code_interpreter_data_source)
         self.interaction_service = InteractionService(self.ai_guesser, self.code_interpreter_data_source)


    def is_data_loaded(self):
         return self._data_loaded

    def set_data_classified(self, status: bool):
         self._data_classified = status

    def is_data_classified(self):
         return self._data_classified


    def execute_command(self, command) -> CLIState:
        """Helper to execute a command and get the next state."""
        # In a real scenario, maybe add logging or error handling here
        return command.execute(self)

    def run(self):
        self._running = True
        while self._running:
            self._current_state.display_menu()
            user_input = input(">>> ")
            self._current_state = self._current_state.handle_input(user_input)
            # If handle_input returned None (like from ExitCommand), loop terminates


    def stop(self):
        self._running = False