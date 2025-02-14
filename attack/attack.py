import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import csv

def generate_calibration_matrix(c, p, q, shift):
    """
    生成校准矩阵M, 维度为c x c
    c: 矩阵大小
    p: 真正频率的概率
    q: 移位后频率的概率
    shift: 移位的位数
    """
    # M = np.zeros((c, c))
    M = lil_matrix((c, c))

    for i in range(c):
        M[i, i] = p  # 对角线填充概率p，对应真实频率的贡献

        # 计算移位后的索引，模c确保在矩阵范围内
        shifted_index = (i + shift) % c
        M[i, shifted_index] = q  # 移位后的概率填充
    return M

def perturb_set_valued_data(item_vectors, epsilon, l):
    """
    对item_vectors进行扰动
    :param item_vectors: 每个订单的01向量表示 (NumPy数组)
    :param epsilon: 隐私预算
    :param l: 循环移位的位数
    :return: 扰动后的集合值数据
    """
    # 获取item_vectors的行列数
    num_orders, num_items = item_vectors.shape
    # print(num_orders)
    # 计算扰动概率 p 和 q
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    q = 1 / (np.exp(epsilon) + 1)

    # 初始化一个空的扰动结果数组
    perturbed_vectors = np.zeros_like(item_vectors)
    count = 0
    # 遍历每个订单 (用户的集合值数据)
    for i in range(num_orders):
        if np.random.rand() < p:  # 以概率p保持整行原数据
            perturbed_vectors[i] = item_vectors[i]
            count=count+1
        else:  # 以概率q将整行数据移位
            perturbed_vectors[i] = np.roll(item_vectors[i], -l)  # 向左移位l位
    # print(count)
    return perturbed_vectors

def estimate_true_frequency(perturbed_vectors, epsilon, l):
    """
    根据扰动后的01向量，计算每个item的真实频率的无偏估计
    :param perturbed_vectors: 扰动后的集合值数据 (NumPy数组)
    :param epsilon: 隐私预算
    :param l: 循环移位参数
    :return: 每个item的真实频率无偏估计
    """
    # 获取perturbed_vectors的行列数
    num_orders, num_items = perturbed_vectors.shape

    # 计算扰动概率 p 和 q
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    q = 1 / (np.exp(epsilon) + 1)

    # 每个 item 被选择的总次数（从扰动数据中汇总）
    perturbed_sums = perturbed_vectors.sum(axis=0)

    # 生成校准矩阵（只需要针对 num_items 大小的矩阵）
    M = generate_calibration_matrix(num_items, p, q, l)

    # 使用矩阵左除来校正频率估计
    f_true = spsolve(M.tocsc(), perturbed_sums)/num_orders

    # 将频率限制在合理范围内 [0, 1]
    f_true = np.clip(f_true, 0, 1)

    return f_true


def calculate_probability(vector, estimated_frequencies):
    """
    根据估计的频率和扰动概率，计算一个给定数据向量出现的概率
    :param vector: 用户的01向量（可能已扰动）
    :param estimated_frequencies: 每个 item 的估计频率
    :return: 向量出现的概率
    """
    # 将输入转换为 numpy 数组
    vector = np.asarray(vector)
    estimated_frequencies = np.asarray(estimated_frequencies)

    # 检查输入维度是否一致
    if vector.shape != estimated_frequencies.shape:
        raise ValueError("向量和估计频率的维度必须相同。")

    # 选取 vector 中等于 1 的位置，对应的频率相乘
    probability = np.prod(estimated_frequencies[vector == 1])

    return probability

def infer_original_data(perturbed_vectors, estimated_frequencies, epsilon, l, t):
    """
    根据用户扰动数据和估计频率，推测每个用户的原始数据
    并标记哪些用户数据为空
    :param perturbed_vectors: 所有用户的扰动结果 (NumPy数组)
    :param estimated_frequencies: 每个item的估计频率 (NumPy数组)
    :param epsilon: 隐私预算
    :param l: 循环移位的位数
    :param t: 阈值，决定是否标记为空用户
    :return: 推测出的用户原始数据, 用户得分, 空用户数量, 空用户标记
    """
    num_users, num_items = perturbed_vectors.shape
    inferred_data = np.zeros_like(perturbed_vectors)

    score = np.zeros((num_users,))
    null_users = np.zeros((num_users,), dtype=bool)  # 标记空用户

    # 遍历每个用户
    for i in range(num_users):
        current_vector = perturbed_vectors[i]
        current_probability = calculate_probability(current_vector, estimated_frequencies)

        shifted_vector = np.roll(current_vector, l)
        shifted_probability = calculate_probability(shifted_vector, estimated_frequencies)

        denominator = shifted_probability + current_probability
        score[i] = max(current_probability, shifted_probability) / denominator if denominator else 0

        if score[i] > t:
            if shifted_probability > current_probability:
                inferred_data[i] = shifted_vector
            else:
                inferred_data[i] = current_vector
        else:
            inferred_data[i] = current_vector
            null_users[i] = True  # 标记为空用户

    null_count = null_users.sum()

    return inferred_data, score, null_count, null_users


def calculate_non_empty_accuracy(inferred_data, original_data, null_users):
    """
    计算非空用户的推测准确性
    :param inferred_data: 推测出的原始数据 (NumPy数组)
    :param original_data: 真实的原始数据 (NumPy数组)
    :param null_users: 空用户标记 (布尔数组)
    :return: 非空用户的准确性
    """
    non_empty_users = ~null_users  # 取非空用户
    if non_empty_users.sum() == 0:
        return 0  # 如果没有非空用户，准确性为0
    non_empty_accuracy = np.mean(np.all(inferred_data[non_empty_users] == original_data[non_empty_users], axis=1))
    return non_empty_accuracy


def calculate_accuracy(inferred_data, original_data, perturbed_data):
    """
    计算推测数据与原始数据、扰动数据的准确性
    :param inferred_data: 推测出的原始数据 (NumPy数组)
    :param original_data: 真实的原始数据 (NumPy数组)
    :param perturbed_data: 扰动后的数据 (NumPy数组)
    :return: 原始数据准确性、扰动数据准确性
    """
    # 比较推测数据与原始数据的相似度
    original_accuracy = np.mean(np.all(inferred_data == original_data, axis=1))

    # 比较推测数据与扰动数据的相似度
    perturbed_accuracy = np.mean(np.all(inferred_data == perturbed_data, axis=1))

    # 比较原始数据与扰动数据的相似度
    po_accuracy = np.mean(np.all(original_data == perturbed_data, axis=1))

    return original_accuracy, perturbed_accuracy, po_accuracy


from sklearn.metrics import jaccard_score
import numpy as np


def calculate_jaccard_similarity(user_data):
    """
    计算用户间的Jaccard相似性
    :param user_data: 用户数据 (NumPy数组，每行代表一个用户)
    :return: 用户之间的Jaccard相似性矩阵
    """
    num_users = user_data.shape[0]
    similarity_matrix = np.zeros((num_users, num_users))

    for i in range(num_users):
        for j in range(i, num_users):
            similarity = jaccard_score(user_data[i], user_data[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # 对称矩阵

    return similarity_matrix


def calculate_similarity_stats(similarity_matrix):
    """
    计算相似性矩阵的最大值、最小值和均值
    :param similarity_matrix: 用户间的相似性矩阵
    :return: 最大值、最小值和均值
    """
    # 去除对角线元素
    np.fill_diagonal(similarity_matrix, np.nan)

    # 提取非对角线元素的最大值、最小值和均值
    max_similarity = np.nanmax(similarity_matrix)
    min_similarity = np.nanmin(similarity_matrix)
    mean_similarity = np.nanmean(similarity_matrix)

    return max_similarity, min_similarity, mean_similarity

def calculate_mae(estimated_frequencies, original_frequencies):
    """
    计算估计频率和原始频率的MAE（平均绝对误差）
    :param estimated_frequencies: 估计频率
    :param original_frequencies: 原始频率
    :return: MAE
    """
    t=np.abs(estimated_frequencies - original_frequencies)
    mae = np.mean(t)
    return mae

def reverse_calibrate(inferred_data, epsilon, l, num_orders):
    """
    对推测出的原始数据进行反向校准，得到更为精确的频率估计
    :param inferred_data: 推测出的原始数据 (NumPy数组)
    :param epsilon: 隐私预算
    :param l: 循环移位的位数
    :param num_orders: 总的订单数（用户数）
    :return: 校准后的频率估计
    """
    num_items = inferred_data.shape[1]

    # 计算扰动概率 p 和 q
    p = np.exp(epsilon) / (np.exp(epsilon) + 1)
    q = 1 / (np.exp(epsilon) + 1)

    # 每个item被选择的总次数（从推测数据中汇总）
    inferred_sums = inferred_data.sum(axis=0)

    # 生成校准矩阵M
    M = generate_calibration_matrix(num_items, p, q, l)

    # 使用矩阵左除来校正频率估计
    f_true_calibrated = spsolve(M.tocsc(), inferred_sums) / num_orders

    # 将频率限制在合理范围内 [0, 1]
    f_true_calibrated = np.clip(f_true_calibrated, 0, 1)

    return f_true_calibrated

def plot_precision_null_rate_curve(precision_values, null_rate_values, epsilon, dataset_name):
    """
    绘制精确度与空用户率的关系曲线
    """
    plt.figure(figsize=(8, 6))
    plt.plot(null_rate_values, precision_values, marker='o', linestyle='-', color='b')
    plt.title(f'Precision vs Null Rate (ε={epsilon})')
    plt.xlabel('Null Rate')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.savefig(f"{dataset_name}_precision_null_rate_curve_epsilon_{epsilon}.png")
    plt.show()


def calculate_auc_pn(null_rate_values, precision_values):
    """
    计算精确度-空用户率曲线下面积 (AUC-PN)
    """
    auc_pn = np.trapz(precision_values, null_rate_values)
    return auc_pn
def generate_synthetic_dataset(num_orders, num_items_1, prob_1):
    """
    生成合成数据集，项目集1中的数据根据概率prob_1的二项分布采样，项目集2的数据根据概率prob_2的二项分布采样。
    :param num_orders: 总的订单数
    :param num_items_1: 项目集1的项目数
    :param num_items_2: 项目集2的项目数
    :param prob_1: 项目集1中项目被选择的概率
    :param prob_2: 项目集2中项目被选择的概率
    :return: 生成的合成数据集 (item_vectors_1, item_vectors_2)
    """
    # 项目集1根据概率为prob_1的二项分布生成
    item_vectors_1 = np.random.binomial(1, prob_1, size=(num_orders, num_items_1))

    # 项目集2根据概率为prob_2的二项分布生成
    # item_vectors_2 = np.random.binomial(1, prob_2, size=(num_orders, num_items_2))

    return item_vectors_1

def main():
    epsilon_values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]  # 隐私预算值

    file_path = './dataset/restaurant-1-orders.csv'
    df = pd.read_csv(file_path)
    order_items = df.groupby('Order Number')['Item Name'].apply(list).reset_index()
    mlb = MultiLabelBinarizer()
    item_vectors = mlb.fit_transform(order_items['Item Name'])

    # file_path = './dataset/online_retail_II.xlsx'
    # df = pd.read_excel(file_path)
    # order_items = df.groupby('Customer ID')['Description'].apply(list).reset_index()
    # mlb = MultiLabelBinarizer()
    # item_vectors = mlb.fit_transform(order_items['Description'])


    num_orders, num_items = item_vectors.shape

    l = random.choice(range(1, num_items))  # 随机选取循环移位位数

    # 用于存储每个隐私预算对应的结果
    avg_accuracies = []
    avg_null_rates = []

    # 保存结果到 CSV 文件
    result_file = "experiment_results.csv"
    with open(result_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epsilon", "Average Accuracy", "Average Null Rate"])  # 写入表头

        for epsilon in epsilon_values:
            accuracies = []
            null_rates = []

            for _ in range(20):  # 多次实验取平均
                perturbed_vectors = perturb_set_valued_data(item_vectors, epsilon, l)
                f_true_estimated = estimate_true_frequency(perturbed_vectors, epsilon, l)
                inferred_data, _, null_count, null_users = infer_original_data(
                    perturbed_vectors, f_true_estimated, epsilon, l, t=0.7
                )

                # 计算当前实验的准确率和空用户率
                accuracy = calculate_non_empty_accuracy(inferred_data, item_vectors, null_users)
                null_rate = null_count / num_orders

                accuracies.append(accuracy)
                null_rates.append(null_rate)

            # 记录每个隐私预算的平均值
            avg_accuracy = np.mean(accuracies)
            avg_null_rate = np.mean(null_rates)
            avg_accuracies.append(avg_accuracy)
            avg_null_rates.append(avg_null_rate)

            # 写入当前隐私预算的结果到文件
            writer.writerow([epsilon, avg_accuracy, avg_null_rate])

            # 打印结果
            print(f"隐私预算: {epsilon}")
            print(f"平均准确率: {avg_accuracy * 100:.2f}%")
            print(f"平均空用户率: {avg_null_rate * 100:.2f}%")

    print(f"实验结果已保存到文件：{result_file}")


if __name__ == "__main__":
    main()