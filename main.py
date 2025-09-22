import sys
import os
from similarity import get_similarity


def main():
    """
    主函数: 处理命令行输入、异常检查、计算相似度并输出。

    用法: python main.py [原文绝对路径] [抄袭版绝对路径] [答案绝对路径]
    """
    if len(sys.argv) != 4:
        print("错误: 输入参数不规范。")
        print("用法: python main.py [原文文件] [抄袭版文件] [答案文件]")
        sys.exit(1)

    orig_path, copy_path, answer_path = sys.argv[1:4]

    try:
        # 检查文件存在
        for path in [orig_path, copy_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")
            if not os.path.isfile(path):
                raise ValueError(f"路径不是文件: {path}")

        # 读取文件
        with open(orig_path, 'r', encoding='utf-8') as f1:
            text1 = f1.read()
        with open(copy_path, 'r', encoding='utf-8') as f2:
            text2 = f2.read()

        # 计算相似度
        sim = get_similarity(text1, text2)

        # 写入答案文件
        with open(answer_path, 'w', encoding='utf-8') as f:
            f.write(f"{copy_path}与原文{orig_path}的相似度为{sim:.2f}")

        print(f"相似度计算完成: {sim:.2f} 已写入 {answer_path}")

    except ImportError as e:
        if 'jieba' in str(e):
            print("错误: jieba 库未安装。请运行 'pip install jieba'。")
        if 'sklearn' in str(e):
            print("错误: sklearn 库未安装。请运行 'pip install scikit-learn'。")
        else:
            print(f"导入错误: {e}")
        sys.exit(1)
    except UnicodeDecodeError:
        print("错误: 文件编码不支持 UTF-8。请检查文件编码。")
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
