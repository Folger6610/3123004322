import unittest
from main import main
from io import StringIO
import sys
import os


class TestExceptionHandling(unittest.TestCase):
    @staticmethod
    def run_main_with_args(args):
        """模拟命令行调用main函数，确保所有变量提前初始化"""
        original_argv = sys.argv
        original_stdout = sys.stdout
        exit_code = 0  # 初始化退出码
        captured_output = StringIO()  # 提前初始化，消除警告

        try:
            sys.argv = args
            sys.stdout = captured_output  # 重定向输出到已初始化的对象
            main()
        except SystemExit as e:
            exit_code = e.code
        finally:
            sys.argv = original_argv
            sys.stdout = original_stdout

        return captured_output.getvalue(), exit_code

    def test_file_not_found(self):
        output, exit_code = self.run_main_with_args([
            "main.py", "nonexistent_orig.txt", "nonexistent_copy.txt", "output.txt"
        ])
        self.assertIn("输入文件不存在，请检查路径: nonexistent_orig.txt", output)
        self.assertEqual(exit_code, 1)

    def test_empty_texts(self):
        temp_files = ["empty_orig.txt", "empty_copy.txt", "empty_output.txt"]
        for file in temp_files[:2]:
            with open(file, "w", encoding="utf-8") as f:
                f.write("")
        output, exit_code = self.run_main_with_args([
            "main.py", temp_files[0], temp_files[1], temp_files[2]
        ])
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        self.assertIn("两个文本均为空，无法计算相似度", output)
        self.assertEqual(exit_code, 1)

    def test_encoding_error(self):
        temp_files = ["gbk_orig.txt", "normal_copy.txt", "encoding_output.txt"]
        with open(temp_files[0], "w", encoding="gbk") as f:
            f.write("测试GBK编码文本")
        with open(temp_files[1], "w", encoding="utf-8") as f:
            f.write("正常文本")
        output, exit_code = self.run_main_with_args([
            "main.py", temp_files[0], temp_files[1], temp_files[2]
        ])
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        self.assertIn("编码错误: 文本编码非UTF-8，请检查文件编码", output)
        self.assertEqual(exit_code, 1)

    def test_one_empty_text(self):
        temp_files = ["empty_orig.txt", "normal_copy.txt", "one_empty_output.txt"]
        with open(temp_files[0], "w", encoding="utf-8") as f:
            f.write("")
        with open(temp_files[1], "w", encoding="utf-8") as f:
            f.write("这是正常的抄袭版文本")
        output, exit_code = self.run_main_with_args([
            "main.py", temp_files[0], temp_files[1], temp_files[2]
        ])
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)
        self.assertIn("相似度计算完成: 0.00 已写入 one_empty_output.txt", output)
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
