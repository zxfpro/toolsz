# Mock 商户识别与分类服务 (对应模块2)


# services/classification.py - Mock Classification Service
# Depends on MockCategoryClassifier strategy, MockUserRegistryStorage, MockAIGuesser
# Also acts as a Subject for Observer pattern (notifies on confirmation needed)
from data.storage import MockCategoryClassifier, MockUserRegistryStorage, MockAIGuesser, Subject
from domain.models import Transaction, UserRegistryEntry # Import models

class ClassificationService(Subject):
    def __init__(self, classifier: MockCategoryClassifier, user_registry: MockUserRegistryStorage, ai_guesser: MockAIGuesser):
        super().__init__() # Initialize Subject part
        self._classifier = classifier
        self._user_registry = user_registry
        self._ai_guesser = ai_guesser
        self._transactions_to_confirm = [] # To hold transactions needing user confirmation
        print("ClassificationService initialized.")

    def classify_transactions(self, transactions):
        print(f"ClassificationService: Starting classification for {len(transactions)} transactions...")
        classified_transactions = []
        self._transactions_to_confirm = [] # Reset confirmation list

        for tx in transactions:
            classified_tx = self._classifier.classify(tx, self._user_registry, self._ai_guesser)
            classified_transactions.append(classified_tx)

            # Simulate needing user confirmation for some AI guesses or new merchants
            if classified_tx.category.name == "其他" or ("Mock AI guessed" in classified_tx.category.__repr__() and "星巴克" not in classified_tx.description): # Basic mock logic
                 print(f"ClassificationService: Transaction '{tx.description}' might need user confirmation.")
                 self._transactions_to_confirm.append(classified_tx)
                 # Notify observers that confirmation is needed (e.g., the CLI state)
                 self.notify("needs_confirmation", classified_tx)


        print("ClassificationService: Classification process finished.")
        return classified_transactions

    def get_transactions_needing_confirmation(self):
        return self._transactions_to_confirm

    def confirm_classification(self, transaction: Transaction, category_name: str, specific_info: dict = None):
         print(f"ClassificationService: User confirming classification for '{transaction.description}' as '{category_name}'...")
         # Update the transaction object
         transaction.category = Category(category_name)
         if specific_info:
             transaction.specific_info = specific_info

         # Add/Update user registry based on confirmation (mock logic)
         new_entry = UserRegistryEntry(transaction.description, category_name, specific_info)
         self._user_registry.add_entry(new_entry) # Mock: always adds new entry

         # Remove from pending confirmation list (mock logic)
         # Find and remove the specific transaction instance
         self._transactions_to_confirm = [t for t in self._transactions_to_confirm if t is not transaction]

         print("ClassificationService: Confirmation processed and registry updated (mock).")
         self.notify("classification_confirmed", transaction) # Notify observers
