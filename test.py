import unittest
from similarity import tokenize, levenshtein_distance, get_tfidf_weights, get_similarity


class TestSimilarityComplete(unittest.TestCase):
    # -------------------------- 分词模块 --------------------------
    def test_tokenize_special_chars(self):
        text = "Python编程：2023年&未来！test_case_123"
        result = tokenize(text)
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_tokenize_empty(self):
        self.assertEqual(tokenize(""), [])

    # -------------------------- Levenshtein 距离 --------------------------
    def test_levenshtein_identical(self):
        seq = ["a", "b", "c"]
        self.assertEqual(levenshtein_distance(seq, seq), 0.0)

    def test_levenshtein_different(self):
        seq1 = ["a", "b"]
        seq2 = ["x", "y", "z"]
        dist = levenshtein_distance(seq1, seq2)
        self.assertIsInstance(dist, float)

    def test_levenshtein_long_text(self):
        long_text1 = "机器学习 " * 500
        long_text2 = "机器学习 " * 400 + "深度学习 " * 100
        words1 = tokenize(long_text1)
        words2 = tokenize(long_text2)
        dist = levenshtein_distance(words1, words2)
        self.assertIsInstance(dist, float)
        self.assertTrue(dist > 0)

    # -------------------------- TF-IDF 模块 --------------------------
    def test_tfidf_semantic(self):
        text1 = "人工智能在医疗领域的应用"
        text2 = "AI技术在医学行业的使用"
        _, cos_sim = get_tfidf_weights(text1, text2)
        self.assertIsInstance(cos_sim, float)

    def test_tfidf_no_overlap(self):
        text1 = "苹果 香蕉 橘子"
        text2 = "汽车 飞机 火车"
        _, cos_sim = get_tfidf_weights(text1, text2)
        self.assertIsInstance(cos_sim, float)

    def test_tfidf_dynamic_features(self):
        short_text1 = "人工智能入门"
        short_text2 = "AI 基础教程"
        weights_short, _ = get_tfidf_weights(short_text1, short_text2)
        long_text1 = "自然语言处理是人工智能的重要方向" * 50
        long_text2 = "自然语言处理技术在文本分析中广泛应用" * 40
        weights_long, _ = get_tfidf_weights(long_text1, long_text2)
        self.assertTrue(isinstance(weights_short, dict) and len(weights_short) > 0)
        self.assertTrue(isinstance(weights_long, dict) and len(weights_long) > 0)

    # -------------------------- 最终相似度 --------------------------
    def test_similarity_same(self):
        text = "完全相同的测试文本"
        sim = get_similarity(text, text)
        self.assertEqual(sim, 1.0)

    def test_similarity_empty(self):
        sim = get_similarity("", "有效文本")
        self.assertEqual(sim, 0.0)

    def test_similarity_short_high(self):
        text1 = "我今天去公园散步"
        text2 = "我今天去花园散步"
        sim = get_similarity(text1, text2)
        self.assertTrue(0.5 <= sim <= 0.9)

    def test_similarity_long(self):
        long_text1 = "机器学习是人工智能的核心分支" * 100
        long_text2 = "机器学习技术在各行业的应用" * 90
        sim = get_similarity(long_text1, long_text2)
        self.assertTrue(0.4 <= sim <= 0.8)

    def test_similarity_dynamic_alpha(self):
        short_text1 = "今天天气好"
        short_text2 = "今天天气不错"
        sim_short = get_similarity(short_text1, short_text2)
        long_text1 = "自然语言处理是人工智能的重要方向" * 100
        long_text2 = "自然语言处理技术在文本分析中广泛应用" * 90
        sim_long = get_similarity(long_text1, long_text2)
        self.assertTrue(0.5 <= sim_short <= 0.9)
        self.assertTrue(0.4 <= sim_long <= 0.8)


if __name__ == "__main__":
    unittest.main()
