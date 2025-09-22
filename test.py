import unittest
from similarity import get_similarity, levenshtein_distance, get_tfidf_weights


class TestSimilarity(unittest.TestCase):
    def test_levenshtein(self):
        """测试 Levenshtein 距离，验证简单替换场景"""
        seq1 = ["a", "b"]
        seq2 = ["a", "c"]
        dist = levenshtein_distance(seq1, seq2)
        self.assertEqual(dist, 1, "应为 1 次替换")

    def test_levenshtein_with_weights(self):
        """测试加权 Levenshtein，验证权重影响"""
        seq1 = ["今天", "是"]
        seq2 = ["今天", "星期"]
        weights = {"是": 0.2, "星期": 1.0}
        dist = levenshtein_distance(seq1, seq2, weights)
        self.assertAlmostEqual(dist, 0.6, places=2, msg="替换成本应为 (0.2+1.0)/2=0.6")

    def test_tfidf_weights(self):
        """测试 TF-IDF 权重和余弦相似度"""
        words1 = ["今天", "是", "星期天"]
        words2 = ["今天", "是", "周天"]
        weights, cos_sim = get_tfidf_weights(words1, words2)
        self.assertGreater(cos_sim, 0.75, "余弦相似度应高")
        self.assertIn("天", weights, "关键字权重高")

    def test_similarity(self):
        """测试综合相似度，验证高相似句子"""
        text1 = "今天是星期天，天气晴，今天晚上我要去看电影。"
        text2 = "今天是周天，天气晴朗，我晚上要去看电影。"
        sim = get_similarity(text1, text2)
        self.assertGreater(sim, 0.8, "相似度应大于 0.8")


if __name__ == "__main__":
    unittest.main()
