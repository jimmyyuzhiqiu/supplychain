
import math
import pandas as pd
from dataclasses import dataclass
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# =========================
# 0) 中文字体（解决 matplotlib 中文不显示）
# =========================
def set_cn_font():
    # Linux 环境常见：WenQuanYi Zen Hei；Windows：Microsoft YaHei；macOS：PingFang SC
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Microsoft YaHei", "PingFang SC", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1) 数据结构
# =========================
@dataclass
class Carrier:
    name: str
    unit_cost: float      # 元/件
    start_fee: float      # 元/票
    min_qty: int          # 件
    lead_time: int = 1    # 天（苏州->宁波 工厂：1天）

    def cost(self, qty: int) -> float:
        if qty <= 0:
            return 0.0
        return max(self.start_fee, self.unit_cost * qty)

@dataclass
class Scenario:
    # 时间
    start_date: date
    horizon_days: int

    # 工厂
    factory_init: float
    factory_cap: float
    daily_consumption: float
    inbound_total: int

    # 供应商
    supplier_init: float
    supplier_daily_prod: float
    supplier_prod_end: date  # 生产截止（含当天是否生产由 prod_on() 控制）

    # 承运商
    carriers: list

# =========================
# 2) 生产函数：当天是否有产量
# =========================
def prod_on(scn: Scenario, d: date, inclusive=True) -> float:
    if inclusive:
        return scn.supplier_daily_prod if d <= scn.supplier_prod_end else 0.0
    else:
        return scn.supplier_daily_prod if d < scn.supplier_prod_end else 0.0

# =========================
# 3) 逐日仿真（工厂+供应商）
#    规则：每天顺序 = 供应商生产 -> 发货(起运) -> 工厂到货 -> 工厂消耗
# =========================
def simulate(scn: Scenario, shipments: dict, prod_inclusive=True):
    """
    shipments: {起运日期: [(qty, carrier_name), ...], ...}
    """
    # map carrier_name -> Carrier
    carrier_map = {c.name: c for c in scn.carriers}

    fac = scn.factory_init
    sup = scn.supplier_init
    in_transit = {}  # {到达日期: qty}

    fac_rows, sup_rows = [], []

    stockout_days = 0
    overcap_days = 0
    sup_neg_days = 0

    shipped_total = 0
    total_freight = 0.0

    d = scn.start_date
    for t in range(scn.horizon_days):
        # --- Supplier production (start of day) ---
        prod = prod_on(scn, d, inclusive=prod_inclusive)
        sup_start = sup
        sup += prod

        # --- Departures today ---
        dep_list = shipments.get(d, [])
        ship_qty = sum(q for q, _ in dep_list)

        sup_after_ship = sup - ship_qty
        if sup_after_ship < -1e-9:
            sup_neg_days += 1

        # schedule arrivals
        for q, cname in dep_list:
            car = carrier_map[cname]
            arr = d + timedelta(days=car.lead_time)
            in_transit[arr] = in_transit.get(arr, 0) + q
            total_freight += car.cost(q)

        sup = sup_after_ship
        shipped_total += ship_qty

        # --- Factory arrivals today ---
        arrival = in_transit.pop(d, 0)
        fac_start = fac
        fac_after_arr = fac + arrival

        if fac_after_arr > scn.factory_cap + 1e-9:
            overcap_days += 1

        # --- Factory consumption ---
        fac_end = fac_after_arr - scn.daily_consumption
        if fac_end < -1e-9:
            stockout_days += 1

        fac_rows.append([d, fac_start, arrival, fac_after_arr, scn.daily_consumption, fac_end])
        sup_rows.append([d, sup_start, prod, ship_qty, sup])

        fac = fac_end
        d += timedelta(days=1)

    fac_df = pd.DataFrame(fac_rows, columns=["日期", "期初库存", "到货", "到货后", "消耗", "期末库存"])
    sup_df = pd.DataFrame(sup_rows, columns=["日期", "期初库存", "产量", "发货", "期末库存"])

    summary = {
        "缺货天数": stockout_days,
        "超上限天数": overcap_days,
        "供应商负库存天数": sup_neg_days,
        "总运费": round(total_freight, 2),
        "总发货量": shipped_total,
        "工厂期末库存": round(float(fac), 2),
        "工厂最低期末库存": round(float(fac_df["期末库存"].min()), 2),
        "工厂最高到货后库存": round(float(fac_df["到货后"].max()), 2),
        "供应商最低期末库存": round(float(sup_df["期末库存"].min()), 2),
    }
    return fac_df, sup_df, summary

# =========================
# 4) 计划生成（搜索“最佳方案”）
#    思路：枚举若干“目标库存水平 target_level”，用 (s, S) 规则补货：
#       - 当预测(提前期内)可能缺货时 -> 计划一票到货
#       - 到货后尽量补到 target_level（不超过工厂上限，不超过供应商可用，不超过剩余需求）
#       - 每票自动选择当下该票量最便宜的承运商（满足最小起运量）
#    目标函数：
#       1) 总运费最小
#       2) 若同成本，票数更少
#       3) 若再相同，峰值库存更低（更稳）
# =========================
def best_carrier_for_qty(scn: Scenario, qty: int):
    feasible = []
    for c in scn.carriers:
        if qty >= c.min_qty:
            feasible.append((c.cost(qty), c.name))
    if not feasible:
        return None
    feasible.sort(key=lambda x: x[0])
    return feasible[0][1]

def build_plan_by_target(scn: Scenario, target_level: int, prod_inclusive=True):
    shipments = {}   # {depart_date: [(qty, carrier_name)]}
    in_transit = {}

    fac = scn.factory_init
    sup = scn.supplier_init
    shipped_cum = 0

    d = scn.start_date
    for t in range(scn.horizon_days):
        # supplier production first
        sup += prod_on(scn, d, inclusive=prod_inclusive)

        # arrivals
        if d in in_transit:
            fac += in_transit.pop(d)

        # hard cap (arrivals后立即检查)
        if fac > scn.factory_cap + 1e-9:
            return None  # infeasible

        # consume at end of day
        fac_end = fac - scn.daily_consumption
        if fac_end < -1e-9:
            return None  # stockout

        remaining = scn.inbound_total - shipped_cum

        # 预测：如果不安排“明天到货”，明天末是否会缺货？
        # lead=1：今天起运->明天到货；如果 fac_end - 消耗 < 0 则需要今天发
        need_ship_today = (fac_end - scn.daily_consumption < -1e-9) and (remaining > 0)

        if need_ship_today:
            # 明天到货后库存不能超过上限：fac_end + qty <= cap
            q_cap = scn.factory_cap - fac_end

            # 目标：尽量补到 target_level（明天到货后达到 target_level）
            # 明天到货后 = fac_end + qty
            q_target = max(0, target_level - fac_end)

            # 受限：不超过 cap、不超过供应商库存、不超过剩余需求
            qty = min(q_cap, q_target, sup, remaining)
            qty = int(math.floor(qty + 1e-9))

            # 如果 qty 太小导致没有承运商满足最小起运量，则尝试把 qty 抬到可行的最小起运量（仍受 cap/sup/remaining 限制）
            cname = best_carrier_for_qty(scn, qty)
            if cname is None:
                # 选 min_qty 最小的承运商兜底
                sorted_by_min = sorted(scn.carriers, key=lambda c: c.min_qty)
                picked = None
                for c in sorted_by_min:
                    qty2 = max(qty, c.min_qty)
                    qty2 = min(qty2, int(q_cap), int(sup), int(remaining))
                    if qty2 >= c.min_qty:
                        picked = (qty2, c.name)
                        break
                if picked is None:
                    return None
                qty, cname = picked

            if qty <= 0:
                return None

            # ship
            sup -= qty
            shipped_cum += qty

            shipments.setdefault(d, []).append((qty, cname))
            # schedule arrival
            lead = [c.lead_time for c in scn.carriers if c.name == cname][0]
            arr = d + timedelta(days=lead)
            in_transit[arr] = in_transit.get(arr, 0) + qty

        fac = fac_end
        d += timedelta(days=1)

    # 必须把总需求发满
    if shipped_cum != scn.inbound_total:
        return None

    return shipments

def pick_best_plan(scn: Scenario, targets=None, prod_inclusive=True):
    if targets is None:
        targets = [800, 1000, 1200, 1500, 1800, 2000]  # 可自行加密

    best = None
    best_pack = None

    for tgt in targets:
        sh = build_plan_by_target(scn, target_level=tgt, prod_inclusive=prod_inclusive)
        if sh is None:
            continue
        fac_df, sup_df, summary = simulate(scn, sh, prod_inclusive=prod_inclusive)

        # 硬约束检查
        if summary["缺货天数"] > 0 or summary["超上限天数"] > 0 or summary["供应商负库存天数"] > 0:
            continue

        # 目标：运费最小 -> 票数最少 -> 峰值更低
        freight = summary["总运费"]
        trips = sum(len(v) for v in sh.values())
        peak = summary["工厂最高到货后库存"]

        score = (freight, trips, peak)

        if best is None or score < best_pack:
            best = (sh, fac_df, sup_df, summary, tgt)
            best_pack = score

    return best

# =========================
# 5) 输出系统录入格式 + 画图
# =========================
def export_plan_csv(scn: Scenario, shipments: dict, path="best_plan.csv"):
    carrier_map = {c.name: c for c in scn.carriers}
    rows = []
    dep_dates = sorted(shipments.keys())
    for i, dep in enumerate(dep_dates):
        for qty, cname in shipments[dep]:
            car = carrier_map[cname]
            arr = dep + timedelta(days=car.lead_time)
            # “多少天一趟”：用与下一票起运的间隔（最后一票写空或 0）
            if i < len(dep_dates) - 1:
                interval = (dep_dates[i+1] - dep).days
            else:
                interval = 0
            rows.append({
                "单趟运量(件)": int(qty),
                "起运日期": dep.isoformat(),
                "承运趟数": 1,
                "多少天一趟": interval,
                "到达日期": arr.isoformat(),
                "承运商": cname,
                "运费预算": round(car.cost(int(qty)), 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df

def plot_factory(fac_df: pd.DataFrame, cap: float, path="factory_inventory.png"):
    set_cn_font()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(fac_df["日期"], fac_df["期末库存"], label="工厂期末库存", linewidth=2)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axhline(cap, color="red", linestyle="--", linewidth=1.5, label=f"上限 {cap:.0f}")
    ax1.set_title("宁波工厂库存曲线（羽绒面料）")
    ax1.set_ylabel("库存（件）")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(fac_df["日期"], fac_df["到货"], alpha=0.25, label="到货量")
    ax2.set_ylabel("到货（件）")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def plot_supplier(sup_df: pd.DataFrame, path="supplier_inventory.png"):
    set_cn_font()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(sup_df["日期"], sup_df["期末库存"], label="供应商期末库存", linewidth=2)
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_title("苏州供应商库存/生产/发货（羽绒面料）")
    ax1.set_ylabel("库存（件）")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(sup_df["日期"], sup_df["产量"], alpha=0.25, label="产量")
    ax2.bar(sup_df["日期"], -sup_df["发货"], alpha=0.25, label="发货(负向显示)")
    ax2.set_ylabel("产量/发货（件）")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

# =========================
# 6) 主程序：按你给的参数跑
# =========================
if __name__ == "__main__":
    # 你给的参数（默认按 2025 年）
    scn = Scenario(
        start_date=date(2025, 1, 1),
        horizon_days=int(round(4243.08 / 74.44)),  # 57天
        factory_init=600.0,
        factory_cap=2000.0,
        daily_consumption=74.44,
        inbound_total=3644,

        supplier_init=200.0,
        supplier_daily_prod=80.0,
        supplier_prod_end=date(2025, 2, 13),

        carriers=[
            Carrier("开源陆运", unit_cost=20.16, start_fee=2000, min_qty=100, lead_time=1),
            Carrier("易达快运", unit_cost=17.64, start_fee=6000, min_qty=341, lead_time=1),
            Carrier("城市配送", unit_cost=28.56, start_fee=500, min_qty=18, lead_time=1),
        ]
    )

    best = pick_best_plan(scn, targets=[800, 1000, 1200, 1500, 1800, 2000], prod_inclusive=True)
    if best is None:
        raise RuntimeError("没有找到满足约束的可行方案，请检查参数/日期/最小起运量等。")

    shipments, fac_df, sup_df, summary, tgt = best

    print("===== 最佳方案(搜索得到) =====")
    print("目标库存 target_level =", tgt)
    print(summary)

    # 导出
    plan_df = export_plan_csv(scn, shipments, path="best_plan.csv")
    fac_df.to_csv("factory_daily.csv", index=False, encoding="utf-8-sig")
    sup_df.to_csv("supplier_daily.csv", index=False, encoding="utf-8-sig")

    # 画图
    plot_factory(fac_df, cap=scn.factory_cap, path="factory_inventory.png")
    plot_supplier(sup_df, path="supplier_inventory.png")

    # 打印“系统录入字段”预览（不以表格方式输出代码）
    print("\n===== 系统录入字段预览（前10行）=====")
    print(plan_df.head(10).to_string(index=False))
