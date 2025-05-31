# Mock 自然语言交互处理服务 (对应模块4的灵活交互)
# Depends on MockAIGuesser and MockCodeInterpreterDataSource
class InteractionService:
    def __init__(self, ai_guesser, data_source):
        self._ai_guesser = ai_guesser
        self._data_source = data_source
        print("InteractionService initialized.")

    def handle_natural_language_query(self, query):
        print(f"InteractionService: Handling natural language query: '{query}' (mock)...")
        # This service acts as the bridge:
        # 1. AI Guesser understands the query
        # 2. AI Guesser tells Code Interpreter DataSource what calculation/data is needed
        # 3. Code Interpreter DataSource performs the (mock) data operation
        # 4. AI Guesser formats the result into natural language

        # The MockAIGuesser's answer_query method simulates this entire flow for simplicity
        response = self._ai_guesser.answer_query(query, self._data_source)

        print("InteractionService: Natural language query handled (mock).")
        return response