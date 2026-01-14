
# -*- coding: utf-8 -*-
import os
import math
from dataclasses import dataclass
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")  # ✅ 强制使用无GUI后端，保证 png 能落盘

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) 中文字体（matplotlib 中文不显示修复）
# =========================
def set_cn_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1) 数据结构
# =========================
@dataclass
class Carrier:
    name: str
    unit_cost: float    # 元/件（已是“该线路单价”，不再乘公里）
    start_fee: float    # 元/票
    min_qty: int        # 件
    lead_time: int = 1  # 天

    def cost(self, qty: int) -> float:
        if qty <= 0:
            return 0.0
        return max(self.start_fee, self.unit_cost * qty)


@dataclass
class BaseParams:
    # 时间
    start_date: date
    horizon_days: int
    prod_end: date              # 生产截止（含当天）
    prod_inclusive: bool = True # 02/13 当天是否生产

    # 工厂
    factory_init: float = 600.0
    factory_cap: float = 2000.0
    daily_consumption: float = 74.44
    inbound_total: int = 3644   # 需补货总量（苏州采购量）

    # 供应商（苏州）
    supplier_init: float = 200.0
    supplier_daily_prod: float = 80.0


@dataclass
class StationParams:
    # 宁波火车站库存参数
    init: float = 0.0
    cap: float = 1000.0
    hold_cost: float = 1.0          # 元/件/天（按期末库存）
    overcap_fee: float = 5.0        # 元/件/天（超上限部分）
    allow_overcap: bool = False     # 是否允许超限（允许则收超限费）


# =========================
# 2) 工具函数
# =========================
def script_dir():
    return os.path.dirname(os.path.abspath(__file__))

def prod_on(bp: BaseParams, d: date) -> float:
    if bp.prod_inclusive:
        return bp.supplier_daily_prod if d <= bp.prod_end else 0.0
    else:
        return bp.supplier_daily_prod if d < bp.prod_end else 0.0

def best_carrier_for_qty(carriers, qty: int):
    feasible = []
    for c in carriers:
        if qty >= c.min_qty:
            feasible.append((c.cost(qty), c.name))
    if not feasible:
        return None
    feasible.sort(key=lambda x: x[0])
    return feasible[0][1]

def enforce_min_qty(carriers, qty, q_cap, q_avail, q_need):
    """
    qty 可能小于所有承运商 min_qty，需要抬到某个承运商 min_qty
    """
    qty = int(math.floor(qty + 1e-9))
    cname = best_carrier_for_qty(carriers, qty)
    if cname is not None:
        return qty, cname

    # 兜底：找 min_qty 最小的承运商并抬高
    sorted_by_min = sorted(carriers, key=lambda c: c.min_qty)
    for c in sorted_by_min:
        qty2 = max(qty, c.min_qty)
        qty2 = min(qty2, int(q_cap), int(q_avail), int(q_need))
        if qty2 >= c.min_qty:
            return int(qty2), c.name

    return None, None


# =========================
# 3) 仿真：直达（供应商->工厂）
# =========================
def simulate_direct(bp: BaseParams, carriers_direct, shipments_direct: dict):
    carrier_map = {c.name: c for c in carriers_direct}

    fac = bp.factory_init
    sup = bp.supplier_init
    in_transit = {}

    fac_rows, sup_rows = [], []

    stockout_days = overcap_days = sup_neg_days = 0
    shipped_total = 0
    freight = 0.0

    d = bp.start_date
    for _ in range(bp.horizon_days):
        # supplier produce
        sup_start = sup
        prod = prod_on(bp, d)
        sup += prod

        # ship today
        dep_list = shipments_direct.get(d, [])
        ship_qty = sum(q for q, _ in dep_list)
        sup_after = sup - ship_qty
        if sup_after < -1e-9:
            sup_neg_days += 1

        for q, cname in dep_list:
            c = carrier_map[cname]
            arr = d + timedelta(days=c.lead_time)
            in_transit[arr] = in_transit.get(arr, 0) + q
            freight += c.cost(int(q))

        sup = sup_after
        shipped_total += ship_qty

        # factory arrival
        arr_qty = in_transit.pop(d, 0)
        fac_start = fac
        fac_after = fac + arr_qty
        if fac_after > bp.factory_cap + 1e-9:
            overcap_days += 1

        fac_end = fac_after - bp.daily_consumption
        if fac_end < -1e-9:
            stockout_days += 1

        fac_rows.append([d, fac_start, arr_qty, fac_after, bp.daily_consumption, fac_end])
        sup_rows.append([d, sup_start, prod, ship_qty, sup])

        fac = fac_end
        d += timedelta(days=1)

    fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","到货","到货后","消耗","期末库存"])
    sup_df = pd.DataFrame(sup_rows, columns=["日期","期初库存","产量","发货","期末库存"])

    summary = {
        "缺货天数": stockout_days,
        "超上限天数": overcap_days,
        "供应商负库存天数": sup_neg_days,
        "总运费": round(freight, 2),
        "总成本(含仓储/超限)": round(freight, 2),
        "总发货量": shipped_total,
        "工厂期末库存": round(float(fac_df["期末库存"].iloc[-1]), 2),
        "工厂最低期末库存": round(float(fac_df["期末库存"].min()), 2),
        "工厂最高到货后库存": round(float(fac_df["到货后"].max()), 2),
        "供应商最低期末库存": round(float(sup_df["期末库存"].min()), 2),
    }
    return fac_df, sup_df, summary


# =========================
# 4) 仿真：中转（供应商->站->工厂）
# =========================
def simulate_via_station(bp: BaseParams, st: StationParams,
                         carriers_leg1, carriers_leg2,
                         shipments_leg1: dict, shipments_leg2: dict):
    map1 = {c.name: c for c in carriers_leg1}
    map2 = {c.name: c for c in carriers_leg2}

    fac = bp.factory_init
    sup = bp.supplier_init
    sta = st.init

    in_transit_1 = {}  # 到站
    in_transit_2 = {}  # 到厂

    fac_rows, sup_rows, sta_rows = [], [], []

    stockout_days = overcap_fac_days = sup_neg_days = 0
    overcap_sta_days = 0
    shipped_total_leg1 = shipped_total_leg2 = 0

    freight1 = freight2 = 0.0
    hold_cost_total = 0.0
    overcap_fee_total = 0.0

    d = bp.start_date
    for _ in range(bp.horizon_days):
        # -------- supplier produce
        sup_start = sup
        prod = prod_on(bp, d)
        sup += prod

        # -------- arrivals
        # to station
        arr_sta = in_transit_1.pop(d, 0)
        sta += arr_sta
        # to factory
        arr_fac = in_transit_2.pop(d, 0)
        fac += arr_fac

        # -------- check caps after arrivals
        if fac > bp.factory_cap + 1e-9:
            overcap_fac_days += 1

        if (not st.allow_overcap) and sta > st.cap + 1e-9:
            overcap_sta_days += 1

        # -------- depart station->factory today
        dep2_list = shipments_leg2.get(d, [])
        dep2_qty = sum(q for q, _ in dep2_list)
        sta_after_dep2 = sta - dep2_qty
        if sta_after_dep2 < -1e-9:
            # 站点不允许负库存（视为不可行）
            overcap_sta_days += 1  # 用这个计数也行，或单独记“站点缺货”
        for q, cname in dep2_list:
            c = map2[cname]
            arr = d + timedelta(days=c.lead_time)
            in_transit_2[arr] = in_transit_2.get(arr, 0) + q
            freight2 += c.cost(int(q))
        sta = sta_after_dep2
        shipped_total_leg2 += dep2_qty

        # -------- depart supplier->station today
        dep1_list = shipments_leg1.get(d, [])
        dep1_qty = sum(q for q, _ in dep1_list)
        sup_after = sup - dep1_qty
        if sup_after < -1e-9:
            sup_neg_days += 1
        for q, cname in dep1_list:
            c = map1[cname]
            arr = d + timedelta(days=c.lead_time)
            in_transit_1[arr] = in_transit_1.get(arr, 0) + q
            freight1 += c.cost(int(q))
        sup = sup_after
        shipped_total_leg1 += dep1_qty

        # -------- factory consume (end of day)
        fac_start = fac - arr_fac  # 仅用于记录展示
        fac_after_arr = fac
        fac_end = fac_after_arr - bp.daily_consumption
        if fac_end < -1e-9:
            stockout_days += 1
        fac = fac_end

        # -------- station holding cost on end-of-day inventory
        hold_cost_total += st.hold_cost * max(sta, 0.0)
        if sta > st.cap + 1e-9:
            if st.allow_overcap:
                overcap_fee_total += st.overcap_fee * (sta - st.cap)
            else:
                overcap_sta_days += 1

        # record
        fac_rows.append([d, fac_start, arr_fac, fac_after_arr, bp.daily_consumption, fac_end])
        sup_rows.append([d, sup_start, prod, dep1_qty, sup])
        sta_rows.append([d, sta + dep2_qty - arr_sta, arr_sta, dep2_qty, sta])

        d += timedelta(days=1)

    fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","到货","到货后","消耗","期末库存"])
    sup_df = pd.DataFrame(sup_rows, columns=["日期","期初库存","产量","发货(到站)","期末库存"])
    sta_df = pd.DataFrame(sta_rows, columns=["日期","期初库存","到货(来自苏州)","发货(到工厂)","期末库存"])

    total_cost = freight1 + freight2 + hold_cost_total + overcap_fee_total

    summary = {
        "缺货天数": stockout_days,
        "工厂超上限天数": overcap_fac_days,
        "供应商负库存天数": sup_neg_days,
        "火车站超上限天数": overcap_sta_days,
        "运费(苏州->站)": round(freight1, 2),
        "运费(站->工厂)": round(freight2, 2),
        "堆存费": round(hold_cost_total, 2),
        "超限费(站)": round(overcap_fee_total, 2),
        "总成本(含仓储/超限)": round(total_cost, 2),
        "总发货量(到站)": shipped_total_leg1,
        "总发货量(到厂)": shipped_total_leg2,
        "工厂最低期末库存": round(float(fac_df["期末库存"].min()), 2),
        "工厂最高到货后库存": round(float(fac_df["到货后"].max()), 2),
        "火车站最低期末库存": round(float(sta_df["期末库存"].min()), 2),
        "火车站最高期末库存": round(float(sta_df["期末库存"].max()), 2),
    }
    return fac_df, sup_df, sta_df, summary


# =========================
# 5) 计划生成：直达（用 target_level 搜索）
# =========================
def build_plan_direct(bp: BaseParams, carriers_direct, target_level: int):
    shipments = {}
    in_transit = {}
    fac = bp.factory_init
    sup = bp.supplier_init
    shipped = 0

    d = bp.start_date
    for _ in range(bp.horizon_days):
        # prod
        sup += prod_on(bp, d)
        # arrivals
        if d in in_transit:
            fac += in_transit.pop(d)

        # cap
        if fac > bp.factory_cap + 1e-9:
            return None
        # consume end
        fac_end = fac - bp.daily_consumption
        if fac_end < -1e-9:
            return None

        remaining = bp.inbound_total - shipped

        # need ship today? (lead=1)
        need_ship_today = (fac_end - bp.daily_consumption < -1e-9) and (remaining > 0)
        if need_ship_today:
            q_cap = bp.factory_cap - fac_end
            q_target = max(0, target_level - fac_end)
            qty = min(q_cap, q_target, sup, remaining)

            qty, cname = enforce_min_qty(carriers_direct, qty, q_cap=q_cap, q_avail=sup, q_need=remaining)
            if cname is None or qty <= 0:
                return None

            # ship
            sup -= qty
            shipped += qty
            shipments.setdefault(d, []).append((int(qty), cname))

            lead = [c.lead_time for c in carriers_direct if c.name == cname][0]
            arr = d + timedelta(days=lead)
            in_transit[arr] = in_transit.get(arr, 0) + qty

        fac = fac_end
        d += timedelta(days=1)

    if shipped != bp.inbound_total:
        return None
    return shipments


# =========================
# 6) 计划生成：中转（双 target 搜索）
#     - factory_target: 工厂补到多少
#     - station_target: 站点补到多少（用于保证有货发到工厂）
# =========================
def build_plan_via_station(bp: BaseParams, st: StationParams,
                           carriers_leg1, carriers_leg2,
                           factory_target: int, station_target: int):
    ship1 = {}  # suzhou->station
    ship2 = {}  # station->factory

    fac = bp.factory_init
    sup = bp.supplier_init
    sta = st.init

    in1 = {}  # arrivals to station
    in2 = {}  # arrivals to factory

    shipped_to_factory = 0  # 以“到厂总量=3644”为准

    d = bp.start_date
    for _ in range(bp.horizon_days):
        # 1) supplier produce
        sup += prod_on(bp, d)

        # 2) arrivals
        sta += in1.pop(d, 0)
        fac += in2.pop(d, 0)

        # 3) hard cap checks
        if fac > bp.factory_cap + 1e-9:
            return None, None

        if (not st.allow_overcap) and sta > st.cap + 1e-9:
            return None, None

        remaining = bp.inbound_total - shipped_to_factory

        # 4) decide station->factory shipment today for arrival tomorrow
        # 判断：不安排明天到货，则明天末是否缺货？
        # lead=1：今天从站发，明天到厂
        fac_end_today = fac - bp.daily_consumption
        if fac_end_today < -1e-9:
            return None, None

        need_ship2_today = (fac_end_today - bp.daily_consumption < -1e-9) and (remaining > 0)
        qty2 = 0
        cname2 = None
        if need_ship2_today:
            # 明天到货后目标补到 factory_target
            q_cap = bp.factory_cap - fac_end_today
            q_target = max(0, factory_target - fac_end_today)

            qty2_raw = min(q_cap, q_target, sta, remaining)
            qty2, cname2 = enforce_min_qty(carriers_leg2, qty2_raw, q_cap=q_cap, q_avail=sta, q_need=remaining)
            if cname2 is None or qty2 <= 0:
                return None, None

            # 执行发货（站->厂）
            sta -= qty2
            shipped_to_factory += qty2
            ship2.setdefault(d, []).append((int(qty2), cname2))
            lead2 = [c.lead_time for c in carriers_leg2 if c.name == cname2][0]
            in2[d + timedelta(days=lead2)] = in2.get(d + timedelta(days=lead2), 0) + qty2

        # 5) decide supplier->station shipment today to maintain station_target
        # 目标：站点尽量补到 station_target（考虑明天站点还要发货不？这里用 station_target 做 base-stock）
        # 注意：站点 cap 约束（若不允许超限）
        sta_cap_limit = st.cap if (not st.allow_overcap) else (10**9)

        # 站点“理想补货量”
        q_need1 = max(0, station_target - sta)
        if remaining <= 0:
            q_need1 = 0

        if q_need1 > 0:
            # 不能超过站点容量限制（到站在明天，因此这里用“明天到站后可能超”的粗约束：sta + qty1 <= cap）
            # 为简化：用 sta_cap_limit - sta 作为上限
            q_cap_sta = max(0, sta_cap_limit - sta)
            qty1_raw = min(q_need1, q_cap_sta, sup, remaining)  # remaining 以最终到厂为准，避免多发

            qty1, cname1 = enforce_min_qty(carriers_leg1, qty1_raw, q_cap=q_cap_sta, q_avail=sup, q_need=remaining)
            if cname1 is None or qty1 <= 0:
                # 允许 station_target 设得太高/太低导致不可行：直接判无解
                return None, None

            sup -= qty1
            ship1.setdefault(d, []).append((int(qty1), cname1))
            lead1 = [c.lead_time for c in carriers_leg1 if c.name == cname1][0]
            in1[d + timedelta(days=lead1)] = in1.get(d + timedelta(days=lead1), 0) + qty1

        # 6) end-of-day: factory consume
        fac = fac_end_today

        d += timedelta(days=1)

    if shipped_to_factory != bp.inbound_total:
        return None, None
    return ship1, ship2


# =========================
# 7) 导出系统录入字段 CSV（每条线路各一份）
# =========================
def export_plan_csv(carriers, shipments: dict, filename: str):
    cmap = {c.name: c for c in carriers}
    dep_dates = sorted(shipments.keys())
    rows = []
    for i, dep in enumerate(dep_dates):
        for qty, cname in shipments[dep]:
            c = cmap[cname]
            arr = dep + timedelta(days=c.lead_time)
            interval = (dep_dates[i+1] - dep).days if i < len(dep_dates)-1 else 0
            rows.append({
                "单趟运量(件)": int(qty),
                "起运日期": dep.isoformat(),
                "承运趟数": 1,
                "多少天一趟": interval,
                "到达日期": arr.isoformat(),
                "承运商": cname,
                "运费预算": round(c.cost(int(qty)), 2),
            })
    df = pd.DataFrame(rows)
    out_path = os.path.join(script_dir(), filename)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[输出] {filename} -> {out_path}")
    return df


# =========================
# 8) 画图（保证落盘 + 打印路径）
# =========================
def plot_factory(fac_df, cap, filename):
    set_cn_font()
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(fac_df["日期"], fac_df["期末库存"], linewidth=2, label="工厂期末库存")
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

    out_path = os.path.join(script_dir(), filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[输出] {filename} -> {out_path}")

def plot_supplier(sup_df, filename):
    set_cn_font()
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(sup_df["日期"], sup_df["期末库存"], linewidth=2, label="供应商期末库存")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_title("苏州供应商库存/生产/发货（羽绒面料）")
    ax1.set_ylabel("库存（件）")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    # 产量正向、发货负向显示
    ax2.bar(sup_df["日期"], sup_df["产量"], alpha=0.25, label="产量")
    # 找到发货列名（直达/中转不同）
    ship_col = "发货" if "发货" in sup_df.columns else "发货(到站)"
    ax2.bar(sup_df["日期"], -sup_df[ship_col], alpha=0.25, label="发货(负向显示)")
    ax2.set_ylabel("产量/发货（件）")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    out_path = os.path.join(script_dir(), filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[输出] {filename} -> {out_path}")

def plot_station(sta_df, cap, filename):
    set_cn_font()
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(sta_df["日期"], sta_df["期末库存"], linewidth=2, label="火车站期末库存")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axhline(cap, color="red", linestyle="--", linewidth=1.5, label=f"上限 {cap:.0f}")
    ax1.set_title("宁波火车站库存/到货/发货（羽绒面料）")
    ax1.set_ylabel("库存（件）")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(sta_df["日期"], sta_df["到货(来自苏州)"], alpha=0.25, label="到货(来自苏州)")
    ax2.bar(sta_df["日期"], -sta_df["发货(到工厂)"], alpha=0.25, label="发货(到工厂,负向)")
    ax2.set_ylabel("到货/发货（件）")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    out_path = os.path.join(script_dir(), filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"[输出] {filename} -> {out_path}")


# =========================
# 9) 一键搜索：直达 vs 中转
# =========================
def search_best_direct(bp, carriers_direct, targets):
    best = None
    for tgt in targets:
        sh = build_plan_direct(bp, carriers_direct, target_level=tgt)
        if sh is None:
            continue
        fac_df, sup_df, summary = simulate_direct(bp, carriers_direct, sh)
        if summary["缺货天数"] or summary["超上限天数"] or summary["供应商负库存天数"]:
            continue
        trips = sum(len(v) for v in sh.values())
        score = (summary["总成本(含仓储/超限)"], trips, summary["工厂最高到货后库存"])
        if best is None or score < best["score"]:
            best = {"ship": sh, "fac": fac_df, "sup": sup_df, "summary": summary, "tgt": tgt, "score": score}
    return best

def search_best_via_station(bp, st, carriers1, carriers2, factory_targets, station_targets):
    best = None
    for ft in factory_targets:
        for stt in station_targets:
            sh1, sh2 = build_plan_via_station(bp, st, carriers1, carriers2, factory_target=ft, station_target=stt)
            if sh1 is None:
                continue
            fac_df, sup_df, sta_df, summary = simulate_via_station(bp, st, carriers1, carriers2, sh1, sh2)
            # 约束：工厂不缺货，供应商不负；站点超限按 allow_overcap 决定
            if summary["缺货天数"] > 0 or summary["供应商负库存天数"] > 0:
                continue
            if (not st.allow_overcap) and summary["火车站超上限天数"] > 0:
                continue
            trips = sum(len(v) for v in sh1.values()) + sum(len(v) for v in sh2.values())
            peak = max(summary["工厂最高到货后库存"], summary["火车站最高期末库存"])
            score = (summary["总成本(含仓储/超限)"], trips, peak)
            if best is None or score < best["score"]:
                best = {
                    "ship1": sh1, "ship2": sh2,
                    "fac": fac_df, "sup": sup_df, "sta": sta_df,
                    "summary": summary, "ft": ft, "stt": stt, "score": score
                }
    return best


# =========================
# 10) 主程序
# =========================
if __name__ == "__main__":
    # ——基础参数（你确认：2025，02/13 含当天生产）
    bp = BaseParams(
        start_date=date(2025,1,1),
        horizon_days=int(round(4243.08 / 74.44)),  # 57 天
        prod_end=date(2025,2,13),
        prod_inclusive=True
    )

    # ——直达：苏州->宁波工厂（你给的 UI 报价）
    carriers_direct = [
        Carrier("开源陆运", unit_cost=20.16, start_fee=2000, min_qty=100, lead_time=1),
        Carrier("易达快运", unit_cost=17.64, start_fee=6000, min_qty=341, lead_time=1),
        Carrier("城市配送", unit_cost=28.56, start_fee=500,  min_qty=18,  lead_time=1),
    ]

    # ——中转 Leg1：苏州->宁波火车站
    carriers_leg1 = [
        Carrier("开源陆运", unit_cost=21.89, start_fee=2000, min_qty=92,  lead_time=1),
        Carrier("易达快运", unit_cost=19.15, start_fee=6000, min_qty=314, lead_time=1),
        Carrier("城市配送", unit_cost=31.01, start_fee=500,  min_qty=17,  lead_time=1),
    ]

    # ——中转 Leg2：宁波火车站->宁波工厂
    carriers_leg2 = [
        Carrier("开源陆运", unit_cost=1.96, start_fee=2000, min_qty=1021, lead_time=1),
        Carrier("易达快运", unit_cost=1.71, start_fee=6000, min_qty=3509, lead_time=1),
        Carrier("城市配送", unit_cost=2.77, start_fee=500,  min_qty=181,  lead_time=1),
    ]

    # ——火车站库存参数（你给：堆存1，上限1000，超限5）
    st_no_over = StationParams(init=0.0, cap=1000.0, hold_cost=1.0, overcap_fee=5.0, allow_overcap=False)
    st_allow_over = StationParams(init=0.0, cap=1000.0, hold_cost=1.0, overcap_fee=5.0, allow_overcap=True)

    # 搜索空间（你后面要更“全局”可以加密这里）
    direct_targets = [800, 1000, 1200, 1500, 1800, 2000]
    factory_targets = [800, 1000, 1200, 1500, 1800, 2000]
    station_targets = [200, 400, 600, 800, 1000]  # 站点 base-stock

    print("==== 开始搜索：直达方案 ====")
    best_d = search_best_direct(bp, carriers_direct, direct_targets)
    if best_d is None:
        raise RuntimeError("直达方案无可行解（请检查参数/最小起运量/日期）。")
    print("[直达最佳] target =", best_d["tgt"])
    print(best_d["summary"])

    print("\n==== 开始搜索：中转方案（不允许火车站超限） ====")
    best_vs_no = search_best_via_station(bp, st_no_over, carriers_leg1, carriers_leg2, factory_targets, station_targets)
    print("[中转(不超限)]", "有解" if best_vs_no else "无解")

    print("\n==== 开始搜索：中转方案（允许火车站超限并计费） ====")
    best_vs_yes = search_best_via_station(bp, st_allow_over, carriers_leg1, carriers_leg2, factory_targets, station_targets)
    print("[中转(可超限)]", "有解" if best_vs_yes else "无解")

    # 输出对比结论
    out_txt = os.path.join(script_dir(), "compare_summary.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("===== 直达最佳 =====\n")
        f.write(f"target_level={best_d['tgt']}\n")
        f.write(str(best_d["summary"]) + "\n\n")

        if best_vs_no:
            f.write("===== 中转最佳（不允许超限） =====\n")
            f.write(f"factory_target={best_vs_no['ft']}, station_target={best_vs_no['stt']}\n")
            f.write(str(best_vs_no["summary"]) + "\n\n")
        else:
            f.write("===== 中转（不允许超限）无可行解 =====\n\n")

        if best_vs_yes:
            f.write("===== 中转最佳（允许超限并计费） =====\n")
            f.write(f"factory_target={best_vs_yes['ft']}, station_target={best_vs_yes['stt']}\n")
            f.write(str(best_vs_yes["summary"]) + "\n\n")
        else:
            f.write("===== 中转（允许超限）无可行解 =====\n\n")

        # 简单推荐：选成本最低的可行方案
        candidates = [("直达", best_d["summary"]["总成本(含仓储/超限)"])]
        if best_vs_no:
            candidates.append(("中转不超限", best_vs_no["summary"]["总成本(含仓储/超限)"]))
        if best_vs_yes:
            candidates.append(("中转可超限", best_vs_yes["summary"]["总成本(含仓储/超限)"]))
        candidates.sort(key=lambda x: x[1])
        f.write("===== 推荐（按总成本最低） =====\n")
        f.write(str(candidates[0]) + "\n")

    print(f"\n[输出] compare_summary.txt -> {out_txt}")

    # ——导出直达 CSV + 图
    export_plan_csv(carriers_direct, best_d["ship"], "plan_direct.csv")
    best_d["fac"].to_csv(os.path.join(script_dir(), "factory_daily_direct.csv"), index=False, encoding="utf-8-sig")
    best_d["sup"].to_csv(os.path.join(script_dir(), "supplier_daily_direct.csv"), index=False, encoding="utf-8-sig")
    plot_factory(best_d["fac"], bp.factory_cap, "factory_direct.png")
    plot_supplier(best_d["sup"], "supplier_direct.png")

    # ——如果中转有解，导出中转 CSV + 图（优先用“不超限”的最佳；没有就用“可超限”的最佳）
    chosen_vs = best_vs_no if best_vs_no else best_vs_yes
    if chosen_vs:
        export_plan_csv(carriers_leg1, chosen_vs["ship1"], "plan_via_station_leg1.csv")
        export_plan_csv(carriers_leg2, chosen_vs["ship2"], "plan_via_station_leg2.csv")

        chosen_vs["fac"].to_csv(os.path.join(script_dir(), "factory_daily_via_station.csv"), index=False, encoding="utf-8-sig")
        chosen_vs["sup"].to_csv(os.path.join(script_dir(), "supplier_daily_via_station.csv"), index=False, encoding="utf-8-sig")
        chosen_vs["sta"].to_csv(os.path.join(script_dir(), "station_daily_via_station.csv"), index=False, encoding="utf-8-sig")

        plot_factory(chosen_vs["fac"], bp.factory_cap, "factory_via_station.png")
        plot_supplier(chosen_vs["sup"], "supplier_via_station.png")
        plot_station(chosen_vs["sta"], cap=(st_no_over.cap if not chosen_vs["summary"].get("超限费(站)",0) else st_allow_over.cap),
                     filename="station_via_station.png")

    print("\n==== 完成：所有文件已输出到脚本所在目录 ====")
