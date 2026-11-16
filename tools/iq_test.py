#!/usr/bin/env python3
"""IQ/quality test against a running Strata server.

    python3 tools/iq_test.py [--port 1234] [--think] [--key KEY]

Sections: math (GSM8K-style), logic, knowledge, Chinese, code (executed).
Prints one line per case and the score per section.
"""
import argparse
import http.client
import json
import re

MATH = [
    ("一个班有 28 个学生，其中 3/4 是女生，女生有多少人？只给数字。", "21"),
    ("小明有 15 个苹果，给了小红 4 个，又买了 8 个，现在有多少个？只给数字。", "19"),
    ("一辆车 2 小时行驶 120 公里，速度是多少 km/h？只给数字。", "60"),
    ("3 的 4 次方是多少？只给数字。", "81"),
    ("一个长方形长 8 宽 5，面积是多少？只给数字。", "40"),
    ("100 减去 37 再乘以 2 等于多少？只给数字。", "126"),
    ("如果 5 个工人 5 小时做 5 个零件，10 个工人 10 小时做多少个？只给数字。", "20"),
    ("数列 2, 6, 12, 20, 30, 下一个是多少？只给数字。", "42"),
    ("一打鸡蛋是 12 个，3 打半是多少个？只给数字。", "42"),
    ("一个数加上它的 1/4 等于 25，这个数是多少？只给数字。", "20"),
    ("鸡兔同笼，共 35 个头，94 只脚，鸡有多少只？只给数字。", "23"),
    ("一件商品先涨价 20%，再降价 20%，最终价格是原价的百分之多少？只给数字。", "96"),
]

LOGIC = [
    ("所有猫都是动物，有些动物会飞。能否推出有些猫会飞？只回答能或不能。", "不能"),
    ("如果今天下雨，比赛取消。比赛没有取消，今天下雨了吗？只回答是或否。", "否"),
    ("甲比乙高，乙比丙高，谁最矮？只给人名。", "丙"),
]

KNOW = [
    ("法国首都是哪里？只给地名。", "巴黎"),
    ("谁写了《红楼梦》？只给人名。", "曹雪芹"),
    ("第一次世界大战哪一年结束？只给年份。", "1918"),
    ("HTTP 404 状态码是什么意思？一句话。", "未找到"),
    ("Python 的创造者是谁？只给人名。", "吉多"),
    ("中国的首都是哪里？只给地名。", "北京"),
    ("水的化学式是什么？", "H2O"),
    ("光速约多少 m/s？只给数字。", "299792458"),
]

CHINESE = [
    ("“画蛇添足”是什么意思？一句话。", "多余"),
    ("把“我今天很开心”翻译成英文，只给英文。", "happy"),
    ("用“虽然…但是…”造一个句子。", "但是"),
]

CODE = [
    ("写一个 Python 函数 is_prime(n)，返回 n 是否为质数。只输出代码块。",
     "def is_prime(n):\n    if n < 2: return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0: return False\n        i += 1\n    return True\n",
     "is_prime",
     [(2, True), (1, False), (17, True), (18, False), (97, True)]),
    ("写一个 Python 函数 reverse_words(s)，反转句子中单词的顺序（单词内部不变）。只输出代码块。",
     "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
     "reverse_words",
     [("hello world", "world hello"), ("a b c", "c b a")]),
]


def ask(port, key, q, maxtok=256, think=False):
    body = json.dumps({"model": "qwen3.8-next-flash", "messages": [{"role": "user", "content": q}],
                       "max_tokens": maxtok, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": think}})
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/chat/completions", body,
              {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    d = json.loads(c.getresponse().read())
    return d["choices"][0]["message"].get("content") or ""


def norm(s):
    return re.sub(r"[\s,，。._]", "", s).lower()


def run_section(name, cases, port, key, think, maxtok=256):
    ok = 0
    for q, exp in cases:
        a = ask(port, key, q, maxtok, think).strip().replace("\n", " ")
        good = norm(exp) in norm(a)
        ok += good
        print(("OK  " if good else "MISS"), name, "|", q[:30], "->", a[:46], flush=True)
    print(f"== {name}: {ok}/{len(cases)}", flush=True)
    return ok, len(cases)


def run_code(port, key, think):
    ok = 0
    for q, _ref, fname, tests in CODE:
        a = ask(port, key, q, 512, think)
        m = re.search(r"```(?:python)?\s*(.*?)```", a, re.S)
        code = m.group(1) if m else a
        ns = {}
        try:
            exec(code, ns)
            f = ns.get(fname)
            good = f is not None and all(f(*t[:-1]) == t[-1] if not isinstance(t[-1], bool) else f(*t[:-1]) == t[-1] for t in tests)
        except Exception as e:
            good = False
        ok += good
        print(("OK  " if good else "MISS"), "code |", q[:28], flush=True)
    print(f"== code: {ok}/{len(CODE)}", flush=True)
    return ok, len(CODE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1234)
    ap.add_argument("--key", default="29306136ed566e2ca92839fdbd90e337")
    ap.add_argument("--think", action="store_true")
    a = ap.parse_args()
    tot = n = 0
    for name, cases, mt in (("math", MATH, 256), ("logic", LOGIC, 128), ("know", KNOW, 128), ("zh", CHINESE, 128)):
        o, c = run_section(name, cases, a.port, a.key, a.think, mt)
        tot += o
        n += c
    o, c = run_code(a.port, a.key, a.think)
    tot += o
    n += c
    print(f"TOTAL {tot}/{n} (thinking {'on' if a.think else 'off'})")


if __name__ == "__main__":
    main()
