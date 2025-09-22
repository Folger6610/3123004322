import unittest
import os
import sys
from main import main


class TestMain(unittest.TestCase):
    def test_invalid_args(self):
        """测试参数不规范"""
        sys.argv = ["main.py", "orig.txt"]  # 少参数
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)
        # 无法直接捕获 print，需手动检查输出: "错误: 输入参数不规范"

    def test_file_not_exist(self):
        """测试文件不存在"""
        sys.argv = ["main.py", "nonexistent.txt", "orig_0.8_add.txt", "answer.txt"]
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)
        # 预期输出: "文件不存在: nonexistent.txt"

    def test_not_file(self):
        """测试路径不是文件"""
        sys.argv = ["main.py", ".", "orig_0.8_add.txt", "answer.txt"]  # 目录
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)
        # 预期输出: "路径不是文件: ."

    def test_unicode_error(self):
        """测试编码错误"""
        # 创建 GBK 编码文件
        with open("gbk.txt", "w", encoding="gbk") as f:
            f.write("测试")
        sys.argv = ["main.py", "gbk.txt", "orig_0.8_add.txt", "answer.txt"]
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 1)
        # 预期输出: "错误: 文件编码不支持 UTF-8"
        os.remove("gbk.txt")


if __name__ == "__main__":
    unittest.main()