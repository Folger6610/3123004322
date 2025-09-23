import sys
import os
from similarity import get_similarity


def main():
    """
    主函数: 处理命令行输入、异常检查、计算相似度并输出。
    用法: python main.py [原文文件] [抄袭版文件] [答案文件]
    """
    if len(sys.argv) != 4:
        print("错误: 输入参数不规范。")
        print("用法: python main.py [原文文件] [抄袭版文件] [答案文件]")
        sys.exit(1)
    orig_path, copy_path, answer_path = sys.argv[1:4]

    try:
        # 1. 检查文件存在性和类型
        for path in [orig_path, copy_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"输入文件不存在，请检查路径: {path}")
            if not os.path.isfile(path):
                raise ValueError(f"路径不是文件: {path}")

        # 2. 读取文件（UTF-8 编码）
        with open(orig_path, 'r', encoding='utf-8') as f1:
            text1 = f1.read().strip()  # 去除前后空白，避免空字符干扰
        with open(copy_path, 'r', encoding='utf-8') as f2:
            text2 = f2.read().strip()

        # 3. 空文本逻辑调整：两个空→报错，一个空→计算相似度0.0
        if not text1 and not text2:
            raise ValueError("两个文本均为空，无法计算相似度")
        # 单个空文本不报错，继续计算（get_similarity会返回0.0）

        # 4. 计算相似度
        sim = get_similarity(text1, text2)

        # 5. 写入答案文件+输出结果
        result_msg = f"{copy_path}与原文{orig_path}的相似度为{sim:.2f}"
        with open(answer_path, 'w', encoding='utf-8') as f:
            f.write(result_msg)
        print(f"相似度计算完成: {sim:.2f} 已写入 {answer_path}")

    # 6. 异常处理（提示文本匹配测试）
    except ImportError as e:
        if 'jieba' in str(e):
            print("错误: jieba 库未安装。请运行 'pip install jieba'。")
        elif 'sklearn' in str(e):
            print("错误: sklearn 库未安装。请运行 'pip install scikit-learn'。")
        else:
            print(f"导入错误: {e}")
        sys.exit(1)
    except UnicodeDecodeError:
        print("编码错误: 文本编码非UTF-8，请检查文件编码")
        sys.exit(1)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    except ValueError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"未知错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
