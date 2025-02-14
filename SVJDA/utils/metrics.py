import numpy as np

def compute_cooccurrence_error(freq_estimates, true_cooccur):

    domain_size = len(freq_estimates)
    estimated_cooccur = np.outer(freq_estimates, freq_estimates)
    np.fill_diagonal(estimated_cooccur, 0)
    error = np.abs(true_cooccur - estimated_cooccur)
    # error = np.sqrt((true_cooccur - estimated_cooccur)**2)
    avg_error = np.mean(error)

    return avg_error

def calculate_mae(estimated_frequencies, original_frequencies):

    return np.mean(np.abs(estimated_frequencies - original_frequencies))

# def calculate_mse(estimated_frequencies, original_frequencies):
#     # 如果需要，可以将数据转换为float32以节省内存
#     # estimated_frequencies = np.array(estimated_frequencies, dtype=np.float32)
#     # original_frequencies = np.array(original_frequencies, dtype=np.float32)
#
#     # 逐元素计算均方误差，避免创建大矩阵
#     squared_errors = (estimated_frequencies - original_frequencies) ** 2
#     mse = np.mean(squared_errors)
#
#     return mse

def calculate_mse(estimated_frequencies, original_frequencies):

    return np.mean((estimated_frequencies - original_frequencies) ** 2)


def calculate_kl_divergence(estimated_frequencies, original_frequencies):
    estimated_frequencies = np.clip(estimated_frequencies, 1e-10, 1)  # 避免对数零的情况
    original_frequencies = np.clip(original_frequencies, 1e-10, 1)
    return np.sum(estimated_frequencies * np.log(estimated_frequencies / original_frequencies))


def linf_error(v_hat, v):
    return np.max(np.abs(v_hat - v))

def linf_coerror(v_hat, v):
    abs_error = np.abs(v_hat - v)
    # 取绝对误差矩阵的最大值
    linf_error = np.max(abs_error)
    return linf_error


def calculate_f1(estimated_confidence, true_confidence):
    """计算 F1-Measure (F1 Score)"""
    true_set = set(true_confidence.keys())
    estimated_set = set(estimated_confidence.keys())

    P = len(true_set & estimated_set) / len(estimated_set) if estimated_set else 0
    R = len(true_set & estimated_set) / len(true_set) if true_set else 0

    if P + R == 0:
        return 0
    f1 = 2 * P * R / (P + R)
    return f1


def calculate_ncr(estimated_confidence, true_confidence):
    """计算 Normalized Cumulative Rank (NCR)"""
    l = len(estimated_confidence)
    q_true = {w: l-rank+1 for rank, w in enumerate(sorted(true_confidence, key=true_confidence.get, reverse=True), 1)}
    q_estimated = {w: l-rank+1 for rank, w in
                   enumerate(sorted(estimated_confidence, key=estimated_confidence.get, reverse=True), 1)}

    ncr_num = sum(q_true[w] for w in q_true if w in q_estimated)
    ncr_denom = sum(q_true[w] for w in q_true)

    ncr = ncr_num / ncr_denom if ncr_denom != 0 else 0
    return ncr


def calculate_var(estimated_confidence, true_confidence):
    """计算 Variance (VAR)"""
    var = 0
    count = 0
    for w in true_confidence:
        rho_w = true_confidence[w]
        phi_w = estimated_confidence.get(w, 0)
        # phi_w = 0
        var += (rho_w - phi_w) ** 2
        count += 1
    return var / count if count > 0 else 0


def calculate_confidence(estimated_cooccur, estimated_freq, relations):
    """

    参数:
    estimated_cooccur: 共现矩阵 (numpy array)，表示每个关系对的估计共现频率。
    estimated_freq: 频率数组 (numpy array)，表示每个项的估计支持度。
    relations: 关系对集合 (set)，每个元素为一个元组 (x_a, x_b)，表示一个关系对。

    返回:
    包含每个关系置信度的字典 { (x_a, x_b): confidence }。
    """
    confidence_dict = {}

    for w in relations:
        try:
            x_a, x_b = w  # 解包关系对 w
            # 检查是否可以计算置信度，避免除以零
            if estimated_freq[x_a] > 0:
                confidence = estimated_cooccur[x_a, x_b] / estimated_freq[x_a]
            else:
                confidence = 0  # 如果 x_a 的支持度为零，则置信度为零
            confidence_dict[w] = confidence
        except ValueError:
            print(f"Skipping invalid relation {w}: expected a tuple with two elements.")

    return confidence_dict

def calculate_sup(estimated_cooccur, estimated_freq, relations):
    """

    参数:
    estimated_cooccur: 共现矩阵 (numpy array)，表示每个关系对的估计共现频率。
    estimated_freq: 频率数组 (numpy array)，表示每个项的估计支持度。
    relations: 关系对集合 (set)，每个元素为一个元组 (x_a, x_b)，表示一个关系对。

    返回:
    包含每个关系置信度的字典 { (x_a, x_b): confidence }。
    """
    confidence_dict = {}

    for w in relations:
        try:
            x_a, x_b = w  # 解包关系对 w
            # 检查是否可以计算置信度，避免除以零
            confidence = estimated_cooccur[x_a, x_b]
            confidence_dict[w] = confidence
        except ValueError:
            print(f"Skipping invalid relation {w}: expected a tuple with two elements.")

    return confidence_dict