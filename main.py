# 加载Python自带 或通过pip安装的模块
import random
import jieba
import json

# 加载用户自己的模块
from example_module.features import civil, hardcode

# ----------------------------------------
# # 本地调试时使用的路径配置
#inp_path = '/home/wangshuqi/tianchi/securityAI3_submit_demo_v2/benchmark_texts.txt'
#out_path = 'adversarial.txt'
# ----------------------------------------

# ----------------------------------------
# 提交时使用的路径配置（提交时请激活）
inp_path = '/tcdata/benchmark_texts.txt'
out_path = 'adversarial.txt'
# ----------------------------------------

with open(inp_path, 'r') as f:
    inp_lines = f.readlines()

def transform(line):
    """转换一行文本。

    :param line: 对抗攻击前的输入文本
    :type line: str
    :returns: str -- 对抗攻击后的输出文门
    """
    # 修改以下逻辑
    out_line = hardcode(civil(line))
    return out_line

out_lines = [transform(_line) for _line in inp_lines]
target = json.dumps({'text': out_lines}, ensure_ascii=False)

with open(out_path, 'w') as f:
    f.write(target)
