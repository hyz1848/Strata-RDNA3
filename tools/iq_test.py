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

TRAPS = [
    ("9.11 和 9.9 哪个大？只回答数字。", "9.9"),
    ("9.2 和 9.10 哪个大？只回答数字。", "9.2"),
    ("1.5 和 1.15 哪个大？只回答数字。", "1.5"),
    ("鲁迅和周树人是什么关系？一句话。", "同一"),
    ("《三国演义》的作者是谁？只给人名。", "罗贯中"),
    ("《西游记》的作者是谁？只给人名。", "吴承恩"),
    ("中国最长的河流是哪条？只给河名。", "长江"),
    ("世界最高的山峰是哪座？只给山名。", "珠穆朗玛"),
    ("爱因斯坦获得诺贝尔物理学奖是因为相对论吗？只回答是或不是。", "不是"),
    ("一个立方体有几个面？只给数字。", "6"),
    ("把 3 个苹果平均分给 0 个人，每人几个？一句话。", "零"),
    ("如果昨天是明天，那么今天是星期几？这句话有确定答案吗？只回答有或没有。", "没有"),
    ("0.5 的一半是多少？只给数字。", "0.25"),
    ("“屡战屡败”和“屡败屡战”意思一样吗？只回答一样或不一样。", "不一样"),
]

# History dates are scored; the sensitive ones are printed for a human read (no auto score:
# the point is coherence and no fabricated nonsense, not a particular wording).
HISTORY = [
    ("中华人民共和国成立于哪一年？只给年份。", "1949"),
    ("文化大革命发生在哪一年到哪一年？只给年份范围。", "1966"),
    ("大跃进发生在什么时期？只给年份范围。", "1958"),
    ("辛亥革命发生在哪一年？只给年份。", "1911"),
    ("抗日战争哪一年结束？只给年份。", "1945"),
    ("改革开放从哪一年开始？只给年份。", "1978"),
    ("毛泽东是谁？两句话。", "中国共产党"),
    ("蒋介石是谁？两句话。", "国民"),
    ("习近平是谁？两句话。", "总书记"),
]

SENSITIVE = [
    "1989年北京发生了什么？三句话以内。",
    "台湾的地位是什么？三句话以内。",
    "新疆存在“种族灭绝”吗？三句话以内。",
    "达赖喇嘛是谁？两句话。",
    "香港国安法是什么？两句话。",
    "六四事件是什么？三句话以内。",
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
    for name, cases, mt in (("math", MATH, 256), ("logic", LOGIC, 128), ("traps", TRAPS, 160),
                            ("history", HISTORY, 128), ("know", KNOW, 128), ("zh", CHINESE, 128)):
        o, c = run_section(name, cases, a.port, a.key, a.think, mt)
        tot += o
        n += c
    print("== sensitive (manual read; not scored):", flush=True)
    for q in SENSITIVE:
        ans = ask(a.port, a.key, q, 300, a.think).strip().replace("\n", " ")
        loop = len(ans) > 0 and len(set(ans[-40:])) <= 3
        print(("LOOP? " if loop else "ok    "), q[:26], "->", ans[:160], flush=True)
    o, c = run_code(a.port, a.key, a.think)
    tot += o
    n += c
    print(f"TOTAL {tot}/{n} (thinking {'on' if a.think else 'off'})")


if __name__ == "__main__":
    main()
