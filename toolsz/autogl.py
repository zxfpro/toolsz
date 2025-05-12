"""
基于autogluon的自动框架，
dataloader->autogluon
author：zhaoxuefeng
datetime:2021-09-21 20:49:10
"""

from sklearn.metrics import r2_score
from autogluon.tabular import TabularDataset,TabularPredictor


__all__ = {'auto_tabel_train':'自动化训练表格数据'}

def auto_tabel_train(train_path,test_path,model_save_path,label,id):
    """
    用于自动化训练表格类数据
    :param train_path: 训练集文件路径
    :param test_path: 测试集文件路径
    :param model_save_path:模型保存路径
    :param label: 标签
    :param id: id
    :return:
    """
    train_data = TabularDataset(train_path)
    predictor = TabularPredictor(label=label, path=model_save_path).fit(train_data.drop(columns=[id]))
    test_data = TabularDataset(test_path)
    preds = predictor.predict(test_data.drop(columns=[id, label]))
    print(predictor.info())
    re_score = r2_score(test_data[label], preds)
    print(re_score, "<-re_score")

