from concurrent.futures import ThreadPoolExecutor
import numpy as np
import random
import murmurhash
from .duchi import duchi_user_data


def laplace_mechanism(scale):
    """Generate Laplace noise."""
    return np.random.laplace(0, scale)

def clip(value, eta):
    """Clip value within the specified range."""
    return np.clip(value, -eta, eta)

def murmur_hash(seed, value, mod):
    """使用 MurmurHash 生成伪随机哈希值"""
    hash_input = str(seed) + str(value)
    hash_val = murmurhash.hash(hash_input)
    return hash_val % mod

def client_side_algorithm1(v, b, eta, epsilon, delta):
    d = len(v)
    hash_seed_h = random.randint(0, 100000)
    # 使用 MurmurHash 替代之前的 h 的生成方式
    # h = [murmur_hash(hash_seed_h, item, b) for item in range(d)]
    # h = np.array([murmur_hash(hash_seed_h, item, b) for item in range(d)])

    s = np.random.choice([-1, 1], d)  # Hash function for signs
    B = np.zeros(b)
    # Calculate bin sums
    if b == 1:
        h = np.zeros(d, dtype=int)
        B[0] = sum(s * v)
    else:
        h = np.array([murmur_hash(hash_seed_h, item, b) for item in range(d)])
        for l in range(d):
            B[h[l]] += s[l] * v[l]

    # B[h] += s * v
    # Adding Laplace noise
    B = np.clip(B, -eta, eta)
    if delta*b / epsilon >= eta:
        B_noisy = np.array([duchi_user_data(B[j], epsilon/b, eta) for j in range(b)])
    else:
        B_noisy = np.array([B[j] + laplace_mechanism(delta*b / epsilon) for j in range(b)])
    # B_noisy = np.array([duchi_user_data(B[j], epsilon, eta) for j in range(b)])
    # B_noisy = np.array([B[j] + laplace_mechanism(delta * b / epsilon) for j in range(b)])
    return h, s, B_noisy

# def client_side_algorithm(v, b, eta, epsilon, delta):
#     return [client_side_algorithm1(item, b, eta, epsilon, delta) for item in v]

def client_side_algorithm(v, b, eta, epsilon, delta, num_workers=60):
    """Parallelize client-side algorithm for efficiency."""
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(lambda item: client_side_algorithm1(item, b, eta, epsilon, delta), v))
    return results

def server_side_algorithm(client_data, d, n):
    v_hat = np.zeros(d)

    for (h, s, B_noisy) in client_data:
        for x in range(d):
            index = h[x]
            v_hat[x] += s[x] * B_noisy[index]

    v_hat /= n  # Average over clients
    return np.clip(v_hat, -1, 1)

def generate_sparse_vector(d, k):
    """生成稀疏向量，最多有 k 个 1"""
    v = np.zeros(d)
    indices = np.random.choice(d, k, replace=False)
    v[indices] = 1
    return v