import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def levenshtein_distance(seq1, seq2, weights=None):
    """
    计算两个序列（词列表）的加权 Levenshtein 编辑距离（空间优化）。

    参数:
    - seq1: 第一个序列（list of str）。
    - seq2: 第二个序列（list of str）。
    - weights: 词到权重的字典（可选，基于TF-IDF或其他），默认1.0。

    返回:
    - float: 编辑距离（考虑权重的操作数）。
    """
    if weights is None:
        weights = {}
    m, n = len(seq1), len(seq2)
    # 快速近似：仅对长序列采样
    if m > 1000 or n > 1000:
        overlap = len(set(seq1) & set(seq2)) / min(len(set(seq1)), len(set(seq2))) if min(len(seq1),
                                                                                          len(seq2)) > 0 else 0
        if overlap <= 0.8:  # 低重叠率，采样
            sample_ratio = 0.8
            sample_m, sample_n = int(m * sample_ratio), int(n * sample_ratio)
            seq1, seq2 = seq1[:sample_m], seq2[:sample_n]
            m, n = sample_m, sample_n
    # 空间优化：只用两行 DP
    if m > n:
        seq1, seq2, m, n = seq2, seq1, n, m  # 确保 m <= n
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            w1, w2 = seq1[i - 1], seq2[j - 1]
            cost = 0 if w1 == w2 else (weights.get(w1, 1.0) + weights.get(w2, 1.0)) / 2
            curr[j] = min(
                prev[j] + 1,  # 删除
                curr[j - 1] + 1,  # 插入
                prev[j - 1] + cost  # 替换或匹配
            )
        prev, curr = curr, prev  # 滚动数组
    return prev[n]


def get_tfidf_weights(words1, words2):
    """
    计算两个词列表的TF-IDF权重（复用分词结果）。

    参数:
    - words1, words2: 分词后的词列表（list of str）。

    返回:
    - dict: 词到TF-IDF权重的映射。
    - float: TF-IDF向量的余弦相似度。
    """
    if not words1 or not words2:
        return {}, 0.0
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform([' '.join(words1), ' '.join(words2)])
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    feature_names = vectorizer.get_feature_names_out()
    weights = {}
    for idx, word in enumerate(feature_names):
        weights[word] = max(tfidf_matrix[0, idx], tfidf_matrix[1, idx]) if tfidf_matrix.shape[0] > 1 else tfidf_matrix[
            0, idx]
    return weights, cos_sim


def get_similarity(text1, text2, alpha=0.5):
    """
    计算两个文本的相似度（结合加权Levenshtein和TF-IDF余弦相似度）。

    参数:
    - text1, text2: 输入字符串。
    - alpha: Levenshtein和TF-IDF相似度的权重（0到1，Levenshtein权重，TF-IDF为1-alpha）。

    返回:
    - float: 相似度 (0.0 到 1.0)。
    """
    words1 = [w for w in jieba.lcut(text1) if w.strip()]
    words2 = [w for w in jieba.lcut(text2) if w.strip()]
    # 调试：打印分词结果
    # print("Words1:", words1)
    # print("Words2:", words2)
    if not words1 and not words2:
        return 0.0
    weights, cos_sim = get_tfidf_weights(words1, words2)
    if not words1 or not words2:
        return cos_sim
    dist = levenshtein_distance(words1, words2, weights)
    max_len = max(len(words1), len(words2))
    lev_sim = 1 - (dist / max_len) if max_len > 0 else 0.0
    return alpha * lev_sim + (1 - alpha) * cos_sim