import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    order_items = df.groupby('Order Number')['Item Name'].apply(list).reset_index()
    mlb = MultiLabelBinarizer()
    item_vectors = mlb.fit_transform(order_items['Item Name'])
    return item_vectors, mlb.classes_

def load_and_prepare_datax(file_path):
    # df = pd.read_csv(file_path)
    df = pd.read_excel(file_path)

    # 根据订单号对项目进行分组
    # order_items = df.groupby('Order Number')['Item Name'].apply(list).reset_index()
    order_items = df.groupby('Customer ID')['Description'].apply(list).reset_index()

    # 初始化MultiLabelBinarizer将项目名称转换为0-1向量
    mlb = MultiLabelBinarizer()
    # item_vectors = mlb.fit_transform(order_items['Item Name'])
    item_vectors = mlb.fit_transform(order_items['Description'])

    return item_vectors, mlb.classes_