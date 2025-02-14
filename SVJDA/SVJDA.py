import numpy as np
import json
import os
import pickle
from data0.data_loader import load_and_prepare_data, load_and_prepare_datax
from utils.metrics import calculate_mse, linf_error, linf_coerror, calculate_ncr, calculate_var, calculate_sup
from mechanisms.sparese import client_side_algorithm, server_side_algorithm
from utils.utils import reduce2, update_co_occur_estimates, lap_user_data

# ----------------- 辅助函数 -----------------

def save_results_to_file(results, filename):
    """
    将结果转换成可序列化格式并保存到文件
    """
    serializable_results = {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in results.items()
    }
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=4)


def update_tail_with_reporting_set(length_limit, length_distribution_set, value_result):
    """
    根据上报集合更新尾部，调整估计值
    """
    addi_total_item = 0
    length_limit = int(length_limit)
    for i in range(length_limit + 1, len(length_distribution_set)):
        addi_total_item += length_distribution_set[i] * (i - length_limit)
        if length_distribution_set[i] <= 0:
            break
    if addi_total_item > 0:
        total_item = sum(value_result) * length_limit
        ratio = addi_total_item / total_item if total_item else 0
        for i in range(len(value_result)):
            value_result[i] *= (1.0 + ratio)
    return value_result


def read_item_vectors(input_file):
    """
    从pickle文件中读取item_vectors数据
    """
    with open(input_file, 'rb') as f:
        item_vectors = pickle.load(f)
    return item_vectors


def process_and_save_data(input_file, pickle_output_file, limit):
    """
    读取输入数据，处理后保存为pickle格式
    """
    item_vectors = read_item_vectors(input_file)
    item_vectors = item_vectors[:limit]
    n_users, n_items = item_vectors.shape

    # 计算真实频率
    true_freq = np.mean(item_vectors, axis=0)
    # 计算真实共现矩阵
    true_cooccur = np.zeros((n_items, n_items))
    for user_data in item_vectors:
        indices = np.where(user_data == 1)[0]
        for i in indices:
            for j in indices:
                if i != j:
                    true_cooccur[i, j] += 1
    true_cooccur /= n_users

    with open(pickle_output_file, 'wb') as f:
        pickle.dump((item_vectors, true_freq, true_cooccur), f)
    print(f"Data has been saved to {pickle_output_file}")


def process_data_with_limit(input_file, base_pkl_path, limit):
    """
    根据是否存在pkl文件决定加载或生成数据
    """
    user_file_name = f"{os.path.basename(input_file).split('.')[0]}_limit_{limit}"
    pickle_output_file = os.path.join(base_pkl_path, f"{user_file_name}.pkl")
    if os.path.exists(pickle_output_file):
        print(f"File {pickle_output_file} already exists. Loading...")
        with open(pickle_output_file, 'rb') as f:
            item_vectors, true_freq, true_cooccur = pickle.load(f)
        return item_vectors, true_freq, true_cooccur
    else:
        print(f"File {pickle_output_file} not found. Generating...")
        process_and_save_data(input_file, pickle_output_file, limit)
        with open(pickle_output_file, 'rb') as f:
            item_vectors, true_freq, true_cooccur = pickle.load(f)
        return item_vectors, true_freq, true_cooccur


def compute_ldp_params(k, bins=1, log_val=None, divisor=1.0, err=1e-6):
    """
    根据参数计算LDP算法中的eta和delta
    如果log_val不为None，则使用sqrt(k*log(log_val))/divisor计算eta，否则使用sqrt(k)
    """
    if log_val is not None:
        eta = np.sqrt(k * np.log(log_val)) / divisor
    else:
        eta = np.sqrt(k)
    delta = min(2 * eta, 2 * k)
    if (2 * k / bins > np.log(2 * bins / err)):
        delta = min(delta, 3 * np.sqrt(bins * 2 * k * np.log(2 * bins / err)))
    return eta, delta


def compute_length_threshold(estimated_intersections, percentile=0.9, max_iter=20):
    """
    根据累计贡献，计算截断阈值L
    """
    cumulative_sum = 0
    total = sum(estimated_intersections)
    for i, val in enumerate(estimated_intersections):
        if i > max_iter:
            break
        cumulative_sum += val
        if cumulative_sum > percentile * total:
            break
    L = i
    if L == 0:
        L = 1
    return L


def lengths_to_binary_vectors(lengths, vector_length):
    """
    将用户长度（数字）转换为长度为vector_length的二进制向量（前length位置1，其余0）
    """
    binary_vectors = []
    for length in lengths:
        vec = [0] * vector_length
        for i in range(min(int(length), vector_length)):
            vec[i] = 1
        binary_vectors.append(vec)
    return np.array(binary_vectors)


def get_top_k_pairs(matrix, k, tri_offset=1):
    """
    从矩阵的上三角（不含对角线）中选取值最大的k对下标
    """
    n = matrix.shape[0]
    upper_indices = np.triu_indices(n, k=tri_offset)
    upper_values = matrix[upper_indices]
    top_k_indices = np.argsort(upper_values)[-k:]
    return list(zip(upper_indices[0][top_k_indices], upper_indices[1][top_k_indices]))


def run_ldp_estimation(data, num_items, num_users, epsilon, k_val, bins=1, use_log=False, log_val=None, divisor=1.0):
    """
    对给定数据执行客户端扰动和服务端聚合，返回估计结果
    """
    eta, delta = compute_ldp_params(k_val, bins, log_val=log_val, divisor=divisor) if use_log else compute_ldp_params(k_val, bins)
    perturbed_data = client_side_algorithm(data, bins, eta, epsilon, delta)
    estimates = server_side_algorithm(perturbed_data, num_items, num_users)
    return estimates


# ----------------- 主函数 -----------------

def main():
    file_path = './archive/restaurant-2-orders.csv'
    item_vectors, item_classes = load_and_prepare_data(file_path)
    n_users, n_items = item_vectors.shape

    true_freq = np.mean(item_vectors, axis=0)
    true_cooccur = np.zeros((n_items, n_items))
    for user_data in item_vectors:
        indices = np.where(user_data == 1)[0]
        for i in indices:
            for j in indices:
                if i != j:
                    true_cooccur[i, j] += 1
    true_cooccur /= n_users

    # input_file = './archive/ifttt.txt'  # 输入数据文件
    # base_pkl_path = './archive/processed_data'  # 处理后数据保存路径
    # limit = 300000  # 限制加载的数据量
    #
    # # 加载或生成数据
    # item_vectors, true_freq, true_cooccur = process_data_with_limit(input_file, base_pkl_path, limit)
    # n_users, n_items = item_vectors.shape

    epsilon_values = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    num_iterations = 10
    k = 64

    # 从真实共现矩阵中获取前k对关系（上三角部分）
    t_n_items_total = true_cooccur.shape[0]
    t_upper_tri_indices = np.triu_indices(t_n_items_total, k=1)
    t_co_occur_upper = true_cooccur[t_upper_tri_indices]
    t_top_k_joint_indices = np.argsort(t_co_occur_upper)[-k:]
    t_top_k_item_pairs = list(zip(t_upper_tri_indices[0][t_top_k_joint_indices],
                                  t_upper_tri_indices[1][t_top_k_joint_indices]))

    phase1_results = []
    phase2_results = []

    for epsilon in epsilon_values:
        print(f"Running experiment with epsilon = {epsilon}")
        freq_errors = []
        ncr_scores = []
        var_scores = []
        phase1_errors = []
        s_coerror = []
        s_error = []

        for _ in range(num_iterations):
            # 数据分为phase1和phase2
            n_users_phase2 = int(n_users * 0.5)
            n_users_phase1 = n_users - n_users_phase2
            original_data_phase1 = item_vectors[:n_users_phase1, :]
            original_data_phase2 = item_vectors[n_users_phase1:, :]

            if n_users > 30000:
                # 当用户数较多时，直接对phase1数据进行LDP估计
                k_sparese = 1
                bins = 1
                freq_estimates_phase1 = run_ldp_estimation(
                    original_data_phase1, n_items, n_users_phase1, epsilon,
                    k_sparese, bins, use_log=False
                )
            else:
                # 否则，将phase1分为三个部分
                n_users_phase1_1 = int(n_users * 0.5)
                n_users_phase1_2 = int(n_users * 0.1)
                n_users_phase1_3 = n_users - n_users_phase1_1 - n_users_phase1_2

                original_data_phase1_1 = item_vectors[:n_users_phase1_1, :]
                original_data_phase1_2 = item_vectors[n_users_phase1_1:n_users_phase1_1 + n_users_phase1_2, :]
                original_data_phase1_3 = item_vectors[n_users_phase1_1 + n_users_phase1_2:, :]

                k_sparese = 1
                bins = 1
                freq_estimates_phase1 = run_ldp_estimation(
                    original_data_phase1_1, n_items, n_users_phase1_1, epsilon,
                    k_sparese, bins, use_log=False
                )

                # 选择频率估计最高的前2k项
                top_2k = 2 * k
                sorted_items = np.argsort(freq_estimates_phase1)[::-1]
                top2k_items = sorted_items[:top_2k]

                n_users_phase1_subset = n_users_phase1_2
                subset_data_phase1 = original_data_phase1_2[:, top2k_items]
                user_lengths_phase1 = np.sum(subset_data_phase1, axis=1)
                binary_vectors_phase1 = lengths_to_binary_vectors(user_lengths_phase1, n_items)
                estimated_intersections_phase1 = run_ldp_estimation(
                    binary_vectors_phase1, top_2k, n_users_phase1_subset, epsilon,
                    k_sparese, bins, use_log=False
                )

                # 对phase1的第三部分处理（仅保留top2k项）
                original_data_phase1_3 = original_data_phase1_3[:, top2k_items]
                L_phase1 = compute_length_threshold(estimated_intersections_phase1, percentile=0.9, max_iter=20)
                k_sparse = L_phase1
                freq_estimates_phase1_3 = run_ldp_estimation(
                    original_data_phase1_3, top_2k, n_users_phase1_3, epsilon,
                    k_sparse, bins, use_log=False
                )
                # 用更新尾部的函数调整估计结果，length_limit取sqrt(k_sparse)
                length_limit = int(np.sqrt(k_sparse))
                freq_estimates_phase1_3 = update_tail_with_reporting_set(length_limit, estimated_intersections_phase1, freq_estimates_phase1_3)
                freq_estimates_phase1[top2k_items] = freq_estimates_phase1_3

            mae_phase1 = calculate_mse(freq_estimates_phase1, true_freq)
            phase1_errors.append(mae_phase1)
            error = linf_error(freq_estimates_phase1, true_freq)
            s_error.append(error)
            phase1_results.append({"mae": mae_phase1, "linf_error": error})

            # 利用频率估计计算共现估计（外积，并置0对角线）
            co_occur_estimates_phase1 = np.outer(freq_estimates_phase1, freq_estimates_phase1)
            np.fill_diagonal(co_occur_estimates_phase1, 0)
            top_k_item_pairs = get_top_k_pairs(co_occur_estimates_phase1, 2 * k, tri_offset=1)
            selected_item_pairs = top_k_item_pairs
            reduced_item_vectors = reduce2(original_data_phase2, selected_item_pairs)

            n_users_phase21 = n_users_phase2 // 5
            n_users_phase22 = n_users_phase2 - n_users_phase21
            reduced_item_vectors1 = reduced_item_vectors[:n_users_phase21, :]
            reduced_item_vectors2 = reduced_item_vectors[n_users_phase21:, :]

            intersection_sizes_phase2 = np.sum(reduced_item_vectors1, axis=1).astype(int)
            binary_vectors_phase2 = lengths_to_binary_vectors(intersection_sizes_phase2, len(selected_item_pairs))
            k_sparese = 1
            bins = 1
            estimated_intersections_phase2 = run_ldp_estimation(
                binary_vectors_phase2, len(selected_item_pairs), n_users_phase21, epsilon,
                k_sparese, bins, use_log=False
            )
            L_phase2 = compute_length_threshold(estimated_intersections_phase2, percentile=0.9, max_iter=30)
            k_sparese = L_phase2
            estimated_distribution = run_ldp_estimation(
                reduced_item_vectors2, len(selected_item_pairs), n_users_phase22, epsilon,
                k_sparese, bins, use_log=False
            )
            length_limit = int(np.sqrt(k_sparese))
            estimated_distribution = update_tail_with_reporting_set(length_limit, estimated_intersections_phase2, estimated_distribution)
            update_co_estimates = update_co_occur_estimates(estimated_distribution, selected_item_pairs, co_occur_estimates_phase1)

            mse = calculate_mse(update_co_estimates, true_cooccur)
            freq_errors.append(mse)
            coerror = linf_coerror(update_co_estimates, true_cooccur)
            s_coerror.append(coerror)
            phase2_results.append({"mse": mse, "L∞ error": coerror})

            W_t = set(t_top_k_item_pairs)
            n_items_total = update_co_estimates.shape[0]
            upper_tri_indices = np.triu_indices(n_items_total, k=1)
            co_occur_upper = update_co_estimates[upper_tri_indices]
            top_k_joint_indices = np.argsort(co_occur_upper)[-k:]
            W_e = set(zip(upper_tri_indices[0][top_k_joint_indices], upper_tri_indices[1][top_k_joint_indices]))

            sup_estimates = calculate_sup(update_co_estimates, freq_estimates_phase1, W_e)
            true_sup = calculate_sup(true_cooccur, true_freq, W_t)
            ncr_score = calculate_ncr(sup_estimates, true_sup)
            var_score = calculate_var(sup_estimates, true_sup)
            ncr_scores.append(ncr_score)
            var_scores.append(var_score)

        avg_freq_error = np.mean(freq_errors)
        avg_s_coerrors = np.mean(s_coerror)
        ncr_scorea = np.mean(ncr_scores)
        var_scorea = np.mean(var_scores)

        print("\n=== Results ===")
        print("MSE:", avg_freq_error)
        print("L∞ error:", avg_s_coerrors)
        print("NCR Scorea:", ncr_scorea)
        print("VAR Scorea:", var_scorea)

        save_results_to_file({
            "phase1_results": phase1_results,
            "phase2_results": phase2_results,
            "evaluation_metrics": {
                "ncr_score": ncr_scores,
                "var_score": var_scores
            }
        }, 'experiment_results.json')


if __name__ == "__main__":
    main()
