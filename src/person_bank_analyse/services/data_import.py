# Mock 数据导入服务 (对应模块1)

# Depends on MockDataImporter strategy and MockTransactionDataStorage
from data.storage import MockDataImporter, MockTransactionDataStorage
from domain.models import Transaction # Import Transaction model

class DataImportService:
    def __init__(self, importer: MockDataImporter, transaction_storage: MockTransactionDataStorage):
        self._importer = importer
        self._transaction_storage = transaction_storage
        print("DataImportService initialized.")

    def import_from_source(self, source_path):
        print(f"DataImportService: Starting import from '{source_path}'...")
        transactions = self._importer.import_data(source_path)
        self._transaction_storage.save_transactions(transactions)
        print("DataImportService: Import process finished.")
        return len(transactions) # Return count of imported transactions
