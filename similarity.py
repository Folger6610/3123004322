import re
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import time


# -------------------------- 1. 分词：精简逻辑，只保留核心过滤 --------------------------
def tokenize(text):
    """仅过滤空字符和纯标点，避免过度处理导致词数偏差"""
    clean_text = re.sub(r'\s+', '', text.strip())
    # 恢复jieba精确模式（平衡精度与速度，避免搜索模式的冗余切分）
    words = jieba.lcut(clean_text, cut_all=False)
    valid_words = [w.strip() for w in words if w.strip() and not re.match(r'^\W+$', w)]
    return valid_words


# -------------------------- 2. Levenshtein：强制浮点类型+精简计算 --------------------------
def levenshtein_distance(seq1, seq2, weights=None):
    if weights is None:
        weights = {}
    m, n = len(seq1), len(seq2)
    print(f"Levenshtein: seq1_len={m}, seq2_len={n}")
    start_lev = time.time()

    # 核心：强制所有数组为float32，初始值显式转为浮点
    if m > n:
        seq1, seq2, m, n = seq2, seq1, n, m

    # 1. 小序列：直接计算，无冗余逻辑
    if max(m, n) < 1000:
        # 初始化dp_prev：[0.0, 1.0, 2.0, ..., n.0]（纯浮点）
        dp_prev = np.array([np.float32(j) for j in range(n + 1)], dtype=np.float32)
        for i in range(1, m + 1):
            # 初始化dp_curr：[i.0, ?, ?, ...]（纯浮点）
            dp_curr = np.zeros(n + 1, dtype=np.float32)
            dp_curr[0] = np.float32(i)
            w1 = seq1[i - 1]
            w1_w = np.float32(weights.get(w1, 1.0))  # 提前获取权重，减少循环内查询
            for j in range(1, n + 1):
                w2 = seq2[j - 1]
                w2_w = np.float32(weights.get(w2, 1.0))
                # 恢复与badsimilarity一致的成本计算（均值），无额外奖励/惩罚
                cost = np.float32(0.0) if w1 == w2 else (w1_w + w2_w) / np.float32(2.0)
                # 三个选项均为float32，彻底解决类型警告
                delete = dp_prev[j] + np.float32(1.0)
                insert = dp_curr[j - 1] + np.float32(1.0)
                replace = dp_prev[j - 1] + cost
                dp_curr[j] = min(delete, insert, replace)
            dp_prev = dp_curr.copy()
        print(f"Levenshtein耗时: {time.time()-start_lev:.2f}s")
        return float(dp_prev[-1])

    # 2. 长序列：块大小300+块数≤10+相邻±2块，平衡精度与速度
    block_size = 300
    blocks1 = min((m + block_size - 1) // block_size, 10)  # 最多10块
    blocks2 = min((n + block_size - 1) // block_size, 10)
    total_dist = np.float32(0.0)

    for i in range(blocks1):
        start1 = i * block_size
        end1 = min((i + 1) * block_size, m)
        block1 = seq1[start1:end1]
        if not block1:
            continue

        min_block_dist = np.float32(float('inf'))
        # 扩大匹配范围至±2块，捕捉更多重复内容，提升lev_sim
        for j in range(max(0, i - 2), min(blocks2, i + 3)):
            start2 = j * block_size
            end2 = min((j + 1) * block_size, n)
            block2 = seq2[start2:end2]
            if not block2:
                continue

            # 块内计算：同小序列逻辑，纯浮点
            mb, nb = len(block1), len(block2)
            if mb > nb:
                block1, block2, mb, nb = block2, block1, nb, mb
            dp_prev = np.array([np.float32(k) for k in range(nb + 1)], dtype=np.float32)
            for ib in range(1, mb + 1):
                dp_curr = np.zeros(nb + 1, dtype=np.float32)
                dp_curr[0] = np.float32(ib)
                w1 = block1[ib - 1]
                w1_w = np.float32(weights.get(w1, 1.0))
                for jb in range(1, nb + 1):
                    w2 = block2[jb - 1]
                    w2_w = np.float32(weights.get(w2, 1.0))
                    cost = np.float32(0.0) if w1 == w2 else (w1_w + w2_w) / np.float32(2.0)
                    delete = dp_prev[jb] + np.float32(1.0)
                    insert = dp_curr[jb - 1] + np.float32(1.0)
                    replace = dp_prev[jb - 1] + cost
                    dp_curr[jb] = min(delete, insert, replace)
                dp_prev = dp_curr.copy()

            block_dist = float(dp_prev[-1])
            if block_dist < min_block_dist:
                min_block_dist = np.float32(block_dist)

        total_dist += min_block_dist

    # 简化距离修正：按块数平均，避免过度放大差异
    total_dist = total_dist / np.float32(blocks1) if blocks1 > 0 else total_dist
    print(f"Levenshtein耗时: {time.time()-start_lev:.2f}s")
    return float(total_dist)


# -------------------------- 3. TF-IDF：恢复1-gram+提升特征数 --------------------------
def get_tfidf_weights(text1, text2):
    """恢复核心参数，避免冗余逻辑拉低余弦相似度"""
    words1 = tokenize(text1)
    words2 = tokenize(text2)
    if not words1 or not words2:
        return {}, 0.0

    # 动态特征数：短文本80（更多保留关键词），长文本500
    total_words = len(words1) + len(words2)
    max_feat = 80 if total_words < 50 else 500

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x,
        preprocessor=lambda x: x,
        lowercase=False,
        max_features=max_feat,
        token_pattern=None,
        stop_words=None,
        ngram_range=(1, 1)  # 恢复1-gram，避免短语冗余拉低相似度
    )

    tfidf_matrix = vectorizer.fit_transform([' '.join(words1), ' '.join(words2)])
    # 余弦相似度强制浮点
    cos_sim = np.float32(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    feature_names = vectorizer.get_feature_names_out()

    # 权重计算：纯浮点处理
    mean_weights = tfidf_matrix.mean(axis=0)
    # 兼容matrix/array，均转为float32数组
    if hasattr(mean_weights, 'A'):
        mean_weights = mean_weights.A[0].astype(np.float32)
    else:
        mean_weights = mean_weights.flatten().astype(np.float32)
    weights = dict(zip(feature_names, mean_weights))

    return weights, float(cos_sim)


def get_similarity(text1, text2):
    start_total = time.time()
    words1 = tokenize(text1)
    words2 = tokenize(text2)

    # 完全相同文本 → 直接返回1.0
    if words1 == words2:
        return 1.0

    # 原有基础检查逻辑
    if not words1 and not words2:
        print(f"总耗时: {time.time()-start_total:.2f}s")
        return 0.0
    if not words1 or not words2:
        _, cos_sim = get_tfidf_weights(text1, text2)
        print(f"总耗时: {time.time()-start_total:.2f}s")
        return cos_sim

    # 计算Levenshtein、TF-IDF、加权融合
    weights, cos_sim = get_tfidf_weights(text1, text2)
    dist = levenshtein_distance(words1, words2, weights)
    max_len = max(len(words1), len(words2))
    lev_sim = np.float32(1.0) - (np.float32(dist) / np.float32(max_len)) if max_len > 0 else np.float32(0.0)

    alpha = np.float32(0.7) if max_len < 50 else np.float32(0.5)
    scale = np.float32(3.0)
    lev_sim_scaled = 1 / (1 + math.exp(-scale * (float(lev_sim) - 0.5)))
    cos_sim_scaled = 1 / (1 + math.exp(-scale * (cos_sim - 0.5)))

    final_sim = float(alpha * lev_sim_scaled + (1 - alpha) * cos_sim_scaled)
    final_sim = max(0.0, min(1.0, final_sim))

    print(f"Final: Levenshtein_sim={lev_sim_scaled:.2f}, Cosine_sim={cos_sim_scaled:.2f}, α={float(alpha)}, "
          f"Similarity={final_sim:.2f}")
    print(f"总耗时: {time.time()-start_total:.2f}s")
    return final_sim
