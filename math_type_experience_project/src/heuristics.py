from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import sympy as sp

from src.schema import AbstractInfo, ExperienceInfo

BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

TYPE_LIBRARY: Dict[str, Dict[str, List[str] | str]] = {
    "arithmetic_total_cost": {
        "coarse_type": "arithmetic",
        "skills": ["unit-price multiplication", "addition", "word-problem translation"],
        "template": "已知单价与数量，再加上额外费用，求总花费。",
        "summary": "先算主项费用，再把附加项累加。",
        "steps": [
            "识别单价、数量与附加费用。",
            "先算单价乘数量得到主项总价。",
            "再把其他费用加到主项总价上。",
            "检查单位是否一致。",
        ],
        "principles": ["总价 = 单价 × 数量 + 附加费用"],
        "formulas": ["total = unit_price * quantity + extras"],
        "pitfalls": ["漏加附加费用", "把单价和数量直接相加"],
    },
    "ratio_scale": {
        "coarse_type": "arithmetic",
        "skills": ["ratio", "proportional reasoning"],
        "template": "已知比值和其中一部分的实际数量，求另一部分。",
        "summary": "先求每一份的大小，再放缩到目标份数。",
        "steps": [
            "把已知对象对应到比例中的份数。",
            "用实际数量除以份数，得到每一份的值。",
            "乘以目标对象对应的份数。",
        ],
        "principles": ["实际数量与比例份数成正比"],
        "formulas": ["part_value = known / known_ratio", "target = part_value * target_ratio"],
        "pitfalls": ["把比例方向弄反", "没有先求每一份大小"],
    },
    "percentage": {
        "coarse_type": "arithmetic",
        "skills": ["percentage", "decimal conversion"],
        "template": "给定百分比和基数，求对应数值。",
        "summary": "把百分数转成小数，再乘以基数。",
        "steps": [
            "把百分数转换为小数或分数。",
            "用基数乘以该比例。",
            "检查结果是否在合理范围内。",
        ],
        "principles": ["百分数 = 比例", "部分 = 比例 × 总量"],
        "formulas": ["part = percent / 100 * whole"],
        "pitfalls": ["25% 写成 25", "把百分比和基数相加"],
    },
    "average": {
        "coarse_type": "arithmetic",
        "skills": ["summation", "division"],
        "template": "给定若干数字，求平均数。",
        "summary": "先求总和，再除以数据个数。",
        "steps": [
            "把所有数字求和。",
            "统计数字个数。",
            "总和除以个数得到平均数。",
        ],
        "principles": ["平均数 = 总和 / 个数"],
        "formulas": ["mean = sum(values) / len(values)"],
        "pitfalls": ["漏掉某个数", "除数写错"],
    },
    "linear_equation": {
        "coarse_type": "algebra",
        "skills": ["equation setup", "isolation of variable"],
        "template": "一元一次方程，目标是求未知数。",
        "summary": "通过移项、合并同类项和除法把 x 单独留下。",
        "steps": [
            "识别方程左右两边。",
            "通过加减法移项，把含 x 的项放在一侧。",
            "化简并用乘除法把 x 单独留下。",
            "代回原式检查。",
        ],
        "principles": ["等式两边同时进行同一运算仍保持相等"],
        "formulas": ["ax + b = c => x = (c - b) / a"],
        "pitfalls": ["移项时符号错误", "忘记检查解"],
    },
    "rectangle_perimeter": {
        "coarse_type": "geometry",
        "skills": ["perimeter formula", "substitution"],
        "template": "矩形周长问题，已知长和宽求周长。",
        "summary": "套用矩形周长公式并代入长宽。",
        "steps": [
            "识别矩形的长和宽。",
            "应用周长公式 2*(长+宽)。",
            "代入数字并计算。",
        ],
        "principles": ["矩形周长 = 2 × (长 + 宽)"],
        "formulas": ["P = 2 * (l + w)"],
        "pitfalls": ["把周长公式误写成长×宽", "漏乘 2"],
    },
    "rectangle_area": {
        "coarse_type": "geometry",
        "skills": ["area formula", "substitution"],
        "template": "矩形面积问题，已知长和宽求面积。",
        "summary": "套用矩形面积公式并代入长宽。",
        "steps": [
            "识别矩形的长和宽。",
            "应用面积公式 长×宽。",
            "代入数字并计算。",
        ],
        "principles": ["矩形面积 = 长 × 宽"],
        "formulas": ["A = l * w"],
        "pitfalls": ["把面积和周长公式混淆"],
    },
    "triangle_area": {
        "coarse_type": "geometry",
        "skills": ["triangle area formula", "substitution"],
        "template": "三角形面积问题，已知底和高求面积。",
        "summary": "用底乘高再除以 2。",
        "steps": [
            "识别三角形的底和高。",
            "应用面积公式 底×高÷2。",
            "代入数字并计算。",
        ],
        "principles": ["三角形面积 = 底 × 高 / 2"],
        "formulas": ["A = b * h / 2"],
        "pitfalls": ["忘记除以 2", "把边长误当作高"],
    },
    "remainder": {
        "coarse_type": "number_theory",
        "skills": ["division algorithm", "modulo"],
        "template": "整除与余数问题，已知被除数和除数求余数。",
        "summary": "使用整除算法或直接取模。",
        "steps": [
            "确定被除数和除数。",
            "做整数除法或取模。",
            "确认余数小于除数。",
        ],
        "principles": ["a = bq + r，且 0 <= r < b"],
        "formulas": ["r = a mod b"],
        "pitfalls": ["把商当作余数", "余数不小于除数"],
    },
    "combinations": {
        "coarse_type": "combinatorics",
        "skills": ["combination counting", "n choose k"],
        "template": "不计顺序的选取问题。",
        "summary": "如果顺序不重要，使用组合数。",
        "steps": [
            "识别总人数 n 和选取人数 k。",
            "确认顺序不重要，因此使用组合。",
            "计算 C(n,k)。",
        ],
        "principles": ["组合数适用于不计顺序的选取"],
        "formulas": ["C(n,k) = n! / (k! * (n-k)!)"],
        "pitfalls": ["把组合误写成排列", "n 和 k 位置弄反"],
    },
    "gcd": {
        "coarse_type": "number_theory",
        "skills": ["greatest common divisor", "factorization"],
        "template": "求两个整数的最大公因数。",
        "summary": "可用公因数枚举、质因数分解或欧几里得算法。",
        "steps": [
            "识别两个整数。",
            "求它们的最大公因数。",
            "验证该数同时整除两者。",
        ],
        "principles": ["最大公因数是同时整除两个整数的最大正整数"],
        "formulas": ["gcd(a,b)"],
        "pitfalls": ["把 gcd 和 lcm 混淆"],
    },
    "lcm": {
        "coarse_type": "number_theory",
        "skills": ["least common multiple", "factorization"],
        "template": "求两个整数的最小公倍数。",
        "summary": "可用质因数分解或与 gcd 的关系式求 lcm。",
        "steps": [
            "识别两个整数。",
            "求它们的最小公倍数。",
            "验证该数是两者的公倍数。",
        ],
        "principles": ["lcm(a,b) * gcd(a,b) = a * b"],
        "formulas": ["lcm(a,b) = a*b / gcd(a,b)"],
        "pitfalls": ["把 gcd 当成 lcm", "没有取最小的公倍数"],
    },
    "consecutive_integers": {
        "coarse_type": "algebra",
        "skills": ["variable definition", "equation solving"],
        "template": "连续整数建模问题。",
        "summary": "设较小整数为 n，较大整数为 n+1，再列方程。",
        "steps": [
            "设较小整数为 n。",
            "把下一个连续整数表示成 n+1。",
            "根据总和列方程并求解。",
            "检查解是否满足连续性。",
        ],
        "principles": ["连续整数可写成 n, n+1"],
        "formulas": ["n + (n+1) = total"],
        "pitfalls": ["连续整数表示错误", "忘记检查整数性"],
    },
    "general_math": {
        "coarse_type": "general_math",
        "skills": ["problem translation"],
        "template": "一般数学题。",
        "summary": "先抽取已知量与未知量，再选择合适策略。",
        "steps": [
            "抽取已知条件与目标量。",
            "选择匹配的数学工具。",
            "分步求解并检查结果。",
        ],
        "principles": ["先建模，再求解"],
        "formulas": [],
        "pitfalls": ["过早代数化", "忽略单位和题意约束"],
    },
}


def normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    value = str(text).strip()
    boxed_match = BOXED_RE.search(value)
    if boxed_match:
        value = boxed_match.group(1)
    value = value.replace("$", "").replace(",", "")
    value = re.sub(r"\s+", " ", value).strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        number = float(value)
        if number.is_integer():
            return str(int(number))
        return (f"{number:.6f}").rstrip("0").rstrip(".")
    return value.lower()


def _contains_all(text: str, terms: List[str]) -> bool:
    return all(term in text for term in terms)


def abstract_problem(question: str) -> AbstractInfo:
    text = question.lower().strip()
    fine_type = "general_math"

    if "greatest common divisor" in text or "gcd" in text:
        fine_type = "gcd"
    elif "least common multiple" in text or "lcm" in text:
        fine_type = "lcm"
    elif "consecutive integers" in text:
        fine_type = "consecutive_integers"
    elif "solve for x" in text or ("x" in text and "=" in text):
        fine_type = "linear_equation"
    elif _contains_all(text, ["rectangle", "perimeter"]):
        fine_type = "rectangle_perimeter"
    elif _contains_all(text, ["rectangle", "area"]):
        fine_type = "rectangle_area"
    elif _contains_all(text, ["triangle", "area"]):
        fine_type = "triangle_area"
    elif "remainder" in text and "divided by" in text:
        fine_type = "remainder"
    elif ("how many ways" in text or "in how many ways" in text) and "choose" in text:
        fine_type = "combinations"
    elif "%" in text or "percent" in text:
        fine_type = "percentage"
    elif "average of" in text or "mean of" in text:
        fine_type = "average"
    elif "ratio of" in text and "if there are" in text:
        fine_type = "ratio_scale"
    elif ("cost" in text or "sells" in text or "spend" in text) and ("each" in text or "buys" in text):
        fine_type = "arithmetic_total_cost"

    item = TYPE_LIBRARY[fine_type]
    return AbstractInfo(
        coarse_type=str(item["coarse_type"]),
        fine_type=fine_type,
        skills=list(item["skills"]),
        template=str(item["template"]),
        rationale=f"Recognized from lexical/math pattern: {fine_type}",
    )


def generate_experience(abstract_info: AbstractInfo, solution: str = "") -> ExperienceInfo:
    item = TYPE_LIBRARY.get(abstract_info.fine_type, TYPE_LIBRARY["general_math"])
    summary = str(item["summary"])
    if solution:
        first_sentence = solution.split(".")[0].strip()
        if first_sentence:
            summary = f"{summary} 参考解法提示：{first_sentence}。"
    return ExperienceInfo(
        strategy_steps=list(item["steps"]),
        key_principles=list(item["principles"]),
        formulas=list(item["formulas"]),
        pitfalls=list(item["pitfalls"]),
        summary=summary,
    )


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return (f"{value:.6f}").rstrip("0").rstrip(".")


def _extract_numbers(text: str) -> List[float]:
    return [float(x) for x in NUMBER_RE.findall(text)]


def _solve_linear_equation(text: str) -> Optional[str]:
    if "=" not in text or "x" not in text:
        return None
    candidate = text
    match = re.search(r"solve for x[:：]?\s*([^?。!]+)", text)
    if match:
        candidate = match.group(1)
    candidate = candidate.rstrip(".?")
    if "=" not in candidate:
        return None
    left, right = candidate.split("=", 1)
    left = left.replace("^", "**").strip()
    right = right.replace("^", "**").strip()
    x = sp.symbols("x")
    try:
        equation = sp.Eq(sp.sympify(left), sp.sympify(right))
        result = sp.solve(equation, x)
    except Exception:
        return None
    if not result:
        return None
    return normalize_answer(str(result[0]))


def _solve_total_cost(text: str) -> Optional[str]:
    unit = re.search(r"(?:for|costs?)\s*(\d+(?:\.\d+)?)\s*(?:dollars?)?\s*each", text)
    quantity = re.search(r"buys?\s*(\d+(?:\.\d+)?)", text)
    if not unit or not quantity:
        return None
    total = float(unit.group(1)) * float(quantity.group(1))
    extras = 0.0

    more_match = re.search(r"(?:then\s*)?(?:pays?|spends?)\s*(\d+(?:\.\d+)?)\s*more dollars?", text)
    plus_match = re.search(r"plus\s*(\d+(?:\.\d+)?)", text)
    standalone_more = re.search(r"(\d+(?:\.\d+)?)\s*more dollars?", text)

    if more_match:
        extras += float(more_match.group(1))
    elif plus_match:
        extras += float(plus_match.group(1))
    elif standalone_more:
        extras += float(standalone_more.group(1))

    total += extras
    return _format_number(total)


def _solve_ratio(text: str) -> Optional[str]:
    ratio = re.search(r"ratio of (.+?) to (.+?) is (\d+)\s*:\s*(\d+)", text)
    known = re.search(r"if there are (\d+) (.+?),", text)
    asked = re.search(r"how many (.+?) are there", text)
    if not ratio or not known:
        return None

    first_name = ratio.group(1).strip()
    second_name = ratio.group(2).strip()
    a = float(ratio.group(3))
    b = float(ratio.group(4))
    known_value = float(known.group(1))
    known_name = known.group(2).strip()
    asked_name = asked.group(1).strip() if asked else ""

    if first_name in known_name or known_name in first_name:
        return _format_number(known_value / a * b)
    if second_name in known_name or known_name in second_name:
        return _format_number(known_value / b * a)
    if asked_name and (second_name in asked_name or asked_name in second_name):
        return _format_number(known_value / a * b)
    if asked_name and (first_name in asked_name or asked_name in first_name):
        return _format_number(known_value / b * a)
    return _format_number(known_value / a * b)


def _solve_percentage(text: str) -> Optional[str]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    pct = float(match.group(1)) / 100.0
    whole = float(match.group(2))
    return _format_number(pct * whole)


def _solve_average(text: str) -> Optional[str]:
    if "average of" not in text and "mean of" not in text:
        return None
    numbers = _extract_numbers(text)
    if not numbers:
        return None
    return _format_number(sum(numbers) / len(numbers))


def _solve_rectangle_geometry(text: str) -> Optional[str]:
    if "rectangle" not in text:
        return None
    numbers = _extract_numbers(text)
    if len(numbers) < 2:
        return None
    length, width = numbers[0], numbers[1]
    if "perimeter" in text:
        return _format_number(2 * (length + width))
    if "area" in text:
        return _format_number(length * width)
    return None


def _solve_triangle_area(text: str) -> Optional[str]:
    if "triangle" not in text or "area" not in text:
        return None
    numbers = _extract_numbers(text)
    if len(numbers) < 2:
        return None
    base, height = numbers[0], numbers[1]
    return _format_number(base * height / 2.0)


def _solve_remainder(text: str) -> Optional[str]:
    match = re.search(r"remainder when (\d+) is divided by (\d+)", text)
    if not match:
        return None
    dividend = int(match.group(1))
    divisor = int(match.group(2))
    return str(dividend % divisor)


def _solve_combinations(text: str) -> Optional[str]:
    match = re.search(r"choose (\d+) [a-z]+ from (\d+) [a-z]+", text)
    if not match:
        match = re.search(r"choose (\d+) students? from (\d+) students?", text)
    if not match:
        numbers = _extract_numbers(text)
        if len(numbers) >= 2 and "choose" in text:
            k, n = int(numbers[0]), int(numbers[1])
        else:
            return None
    else:
        k, n = int(match.group(1)), int(match.group(2))
    return str(math.comb(n, k))


def _solve_gcd_lcm(text: str) -> Optional[str]:
    numbers = [int(x) for x in NUMBER_RE.findall(text)]
    if len(numbers) < 2:
        return None
    a, b = numbers[0], numbers[1]
    if "greatest common divisor" in text or "gcd" in text:
        return str(math.gcd(a, b))
    if "least common multiple" in text or "lcm" in text:
        return str(abs(a * b) // math.gcd(a, b))
    return None


def _solve_consecutive_integers(text: str) -> Optional[str]:
    if "consecutive integers" not in text or "sum to" not in text:
        return None
    match = re.search(r"sum to (\d+)", text)
    if not match:
        return None
    total = int(match.group(1))
    smaller = (total - 1) / 2
    return _format_number(smaller)


def heuristic_solve_math(question: str) -> Optional[str]:
    text = question.lower().strip()
    for solver in [
        _solve_gcd_lcm,
        _solve_consecutive_integers,
        _solve_linear_equation,
        _solve_rectangle_geometry,
        _solve_triangle_area,
        _solve_remainder,
        _solve_combinations,
        _solve_percentage,
        _solve_average,
        _solve_ratio,
        _solve_total_cost,
    ]:
        answer = solver(text)
        if answer is not None:
            return normalize_answer(answer)
    return None
