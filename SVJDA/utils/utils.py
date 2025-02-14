import numpy as np
from mechanisms.olhell import olh_user_data, estimate_frequencies_olh
from mechanisms.adap import adap_user_data, estimate_frequencies_adap


def reduce2(original_data_phase2,selected_item_pairs):
    original_data_phase2 = np.array(original_data_phase2)
    data_phase2 = np.zeros((original_data_phase2.shape[0], len(selected_item_pairs)))
    count = 0
    for i, j in selected_item_pairs:
        data_phase2[:,count] = original_data_phase2[:,i] & original_data_phase2[:,j]
        count += 1

    return data_phase2


def update_co_occur_estimates(data_phase2, selected_item_pairs, co_occur_estimates_phase1):
    """
    使用 data_phase2 和 selected_item_pairs 更新 co_occur_estimates_phase1。

    参数:
    data_phase2: 2D NumPy 数组，包含需要用来更新的稀疏向量。
    selected_item_pairs: 2D NumPy 数组，包含项目对的索引。
    co_occur_estimates_phase1: 2D NumPy 数组，表示初始的共现估计。

    返回:
    更新后的共现估计。
    """
    # 假设 data_phase2 和 selected_item_pairs 的行数相同
    num_columns = len(selected_item_pairs)

    for i in range(num_columns):
        # 获取 selected_item_pairs 中第 i 列的项目对
        item1_idx,item2_idx = selected_item_pairs[i]
        # item2_idx = selected_item_pairs[1, i]

        # 更新共现估计
        if data_phase2[i]>=0:
            co_occur_estimates_phase1[item1_idx, item2_idx] = data_phase2[i]
            co_occur_estimates_phase1[item2_idx, item1_idx] = data_phase2[i]
        # else:
        #     co_occur_estimates_phase1[item1_idx, item2_idx] = min(data_phase2[i],co_occur_estimates_phase1[item1_idx, item2_idx])
        #     co_occur_estimates_phase1[item2_idx, item1_idx] = co_occur_estimates_phase1[item1_idx, item2_idx]

    return co_occur_estimates_phase1


import numpy as np
import itertools

def construct_candidate_set(freq_estimates, k):
    """
    构建候选项集 IS。

    参数:
    freq_estimates: 1D NumPy 数组，表示每个项的频率估计值。
    k: int，候选项集的数量。

    返回:
    IS: list，包含候选项集的索引。
    """
    # Step 1: 计算归一化频率估计值
    max_freq = np.max(freq_estimates)
    normalized_estimates = 0.9 * freq_estimates / max_freq

    # Step 2: 构建候选项集 IS
    # 对 S' 中的项进行组合，构建候选项集
    top_k_indices = np.argsort(freq_estimates)[-k:]  # 选取前 k 个频繁项的索引
    candidate_itemsets = []

    # 生成项集组合，组合大小在 1 到 log_2(k) 之间
    for subset_size in range(1, int(np.log2(k)) + 1):
        for subset in itertools.combinations(top_k_indices, subset_size):
            # 计算猜测频率，作为候选项集的选择依据
            guessed_frequency = np.prod([normalized_estimates[item] for item in subset])
            candidate_itemsets.append((subset, guessed_frequency))

    # 根据猜测频率对候选项集进行排序，选出猜测频率最高的 2k 个项集
    candidate_itemsets = sorted(candidate_itemsets, key=lambda x: x[1], reverse=True)[:2 * k]
    IS = [itemset[0] for itemset in candidate_itemsets]

    return IS

def mine_frequent_itemsets(candidate_set_IS, user_data, epsilon):
    """
    挖掘频繁项集
    """
    n_users = user_data.shape[0]
    vs_list = []
    for user_vector in user_data:
        # 构建 0/1 比特串，表示用户拥有的项集
        vs = np.zeros(len(candidate_set_IS))
        for i, itemset in enumerate(candidate_set_IS):
            if all(user_vector[item] == 1 for item in itemset):
                vs[i] = 1
        vs_list.append(vs)

    # 获取所有用户的 vs 集合大小并使用 OLH 机制进行扰动
    vs_sizes = [int(np.sum(vs)) for vs in vs_list]
    perturbed_sizes = olh_user_data(np.array(vs_sizes), max(vs_sizes)+1, epsilon, 0)

    # 使用 OLH 机制对 |vs| 进行扰动并聚合估计
    estimated_sizes = estimate_frequencies_olh(perturbed_sizes, max(len(itemset) for itemset in candidate_set_IS) + 1, n_users, epsilon)

    # 找到第 90 个百分位数 L
    cumulative_sum = 0
    threshold = 0.9
    L = max(len(itemset) for itemset in candidate_set_IS)  # 初始化 L
    for i, value in enumerate(estimated_sizes):
        cumulative_sum += value
        if cumulative_sum > threshold:
            L = i
            break

    return L, estimated_sizes, vs_list
def report_final_itemsets(candidate_set_IS, user_data, epsilon, L):
    """
    使用 PSFO 机制对 vs 进行填充和采样，最终报告项集
    """
    perturbed_reports = []
    for user_vector in user_data:
        vs = [itemset for itemset in candidate_set_IS if all(user_vector[item] == 1 for item in itemset)]
        sampled_item = np.random.choice(vs) if vs else None
        if sampled_item:
            perturbed_report = adap_user_data(np.array([sampled_item]), epsilon, L)
            perturbed_reports.append(perturbed_report)

    return perturbed_reports


def lap_user_data(data, epsilon):
    """
    对输入数据应用拉普拉斯机制以添加噪声，确保隐私。

    参数:
    - data: np.array，要添加噪声的数据。
    - epsilon: float，隐私预算参数，值越小隐私保护越强。

    返回:
    - perturbed_data: np.array，加噪后的数据。
    """
    # 计算拉普拉斯分布的噪声尺度，假设灵敏度为 1
    sensitivity = 1
    scale = sensitivity / epsilon

    # 生成与输入数据同形状的拉普拉斯噪声
    noise = np.random.laplace(0, scale, data.shape)

    # 将噪声添加到原数据中
    perturbed_data = data + noise

    return perturbed_data