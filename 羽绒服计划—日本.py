
# -*- coding: utf-8 -*-
import os
import math
from dataclasses import dataclass
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")  # ✅ 保证 PNG 一定能保存（即使无GUI）

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) 中文字体
# =========================
def set_cn_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False

def here(path: str) -> str:
    """输出到脚本所在目录"""
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, path)


# =========================
# 1) 数据结构
# =========================
@dataclass
class Carrier:
    name: str
    lead: int           # 天
    unit_cost: float    # 单价（按件）
    start_fee: float    # 起步价
    min_qty: int        # 最小起运量
    currency: str = "CNY"  # CNY/USD/JPY

    def cost_cny(self, qty: int, fx: dict) -> float:
        if qty <= 0:
            return 0.0
        raw = max(self.start_fee, self.unit_cost * qty)
        return raw * fx[self.currency]


@dataclass
class Route:
    name: str
    legs: list  # [(leg_name, [Carrier, ...]), ...]
    total_lead: int
    min_qty: int  # 该路线整票的最小起运量（由各段最小值决定）

    def pick_cheapest_leg_carriers(self, qty: int, fx: dict):
        """给定票量 qty，逐段选满足 min_qty 的最便宜承运商"""
        picks = []
        total_cost = 0.0
        for leg_name, options in self.legs:
            feas = [c for c in options if qty >= c.min_qty]
            if not feas:
                return None
            best = min(feas, key=lambda c: c.cost_cny(qty, fx))
            picks.append((leg_name, best))
            total_cost += best.cost_cny(qty, fx)
        return picks, total_cost


# =========================
# 2) 需求序列（最后一天自动补齐总量）
# =========================
def build_daily_demand(total: int, daily_avg: int):
    days = math.ceil(total / daily_avg)
    dem = [daily_avg] * days
    dem[-1] = total - daily_avg * (days - 1)
    return dem


# =========================
# 3) 计划器：只满足日本
#    策略（可解释&好调参）：
#    - 海运更便宜：尽量用海运做“长期供给”
#    - 开头海运来不及：用空运做“桥接”，防止首批缺货
#    - 每天滚动预测：确保未来不会缺货；若会缺货，选择能及时到达且成本最低的路线
#    - 同时严格控制日本上限 8000：任何到货日“到货后库存”不得超过上限
#    - 工厂上限 6000：每天发货不能超过当日可用库存（生产后）
# =========================
def plan_japan_only(
    start_date: date,
    demand_total: int,
    demand_daily: int,
    jp_init: int,
    jp_cap: int,
    fac_init: int,
    fac_cap: int,
    fac_prod_daily: int,
    fac_prod_end: date,
    routes: dict,
    fx: dict,
):
    demands = build_daily_demand(demand_total, demand_daily)
    horizon = len(demands)
    end_date = start_date + timedelta(days=horizon - 1)

    # 我们尽量让“总发货量 = demand_total - jp_init”（最终期末接近0，成本更低）
    target_ship_total = max(0, demand_total - jp_init)

    # 状态
    fac = fac_init
    jp = jp_init
    shipped_total = 0

    # 到货计划（日本端）：arrival_date -> qty（允许同日多票累计）
    arrivals = {}

    # 输出：每一票的“路线级”记录（后面会展开成每一段 leg 的录入行）
    route_shipments = []  # dict: {route, depart, qty, arrive, cost}

    # 帮助函数：计算某天到货前、日本库存（包含之前已排到货）
    def jp_inventory_before(day: date) -> float:
        inv = jp_init
        d = start_date
        for dem in demands:
            if d == day:
                break
            inv += arrivals.get(d, 0)
            inv -= dem
            d += timedelta(days=1)
        return inv

    # 帮助函数：该天已有的到货量（用于避免多票叠加导致超上限）
    def jp_existing_arrival(day: date) -> int:
        return int(arrivals.get(day, 0))

    # 逐日滚动
    for t in range(horizon):
        today = start_date + timedelta(days=t)

        # 1) 工厂生产（含上限）
        if today <= fac_prod_end:
            fac = min(fac_cap, fac + fac_prod_daily)

        # 2) 日本到货
        jp += arrivals.get(today, 0)
        if jp > jp_cap + 1e-9:
            raise RuntimeError(f"日本库存超上限：{today} 到货后={jp} > {jp_cap}")

        # 3) 日本消耗
        jp -= demands[t]
        if jp < -1e-9:
            raise RuntimeError(f"日本缺货：{today} 期末库存={jp}")

        # 如果已经发够总量，可以不再排新票（但仍要确保不会缺货；这里目标设计保证不会）
        remaining_to_ship = target_ship_total - shipped_total
        if remaining_to_ship <= 0:
            continue

        # 4) 预测：找未来最早会缺货的那一天（在当前已排到货基础上）
        #    我们只用“未来 max_lead 天”窗口做滚动修复，避免排得过远导致上限冲突
        max_lead = max(r.total_lead for r in routes.values())
        lookahead = min(max_lead + 2, horizon - t - 1)

        # 预测库存轨迹（从“今天期末 jp”开始往后推）
        proj = jp
        stockout_day = None
        for k in range(1, lookahead + 1):
            d = today + timedelta(days=k)
            idx = t + k
            proj += arrivals.get(d, 0)
            proj -= demands[idx]
            if proj < -1e-9:
                stockout_day = d
                break

        if stockout_day is None:
            # 短窗口内不会缺货，那就优先“提前排海运”来降低未来空运需求（只要不引发上限）
            # 试着排一票海运（如果到货日不超上限且工厂库存够）
            r = routes["SEA"]
            depart = today
            arrive = depart + timedelta(days=r.total_lead)
            if arrive <= end_date:
                inv_before = jp_inventory_before(arrive)
                existing = jp_existing_arrival(arrive)
                # 到货当天的最大可加量：cap - (到货前库存 + 当天已排到货)
                q_max = math.floor(jp_cap - (inv_before + existing))
                q = min(q_max, remaining_to_ship, fac)

                # 每天允许多票：如果 q 不够大但仍能凑到 min_qty，则可以；否则跳过
                q = int(math.floor(q))
                if q >= r.min_qty:
                    picks_cost = r.pick_cheapest_leg_carriers(q, fx)
                    if picks_cost is not None:
                        picks, cost = picks_cost
                        # 执行
                        fac -= q
                        shipped_total += q
                        arrivals[arrive] = arrivals.get(arrive, 0) + q
                        route_shipments.append({
                            "route": r.name,
                            "depart": depart,
                            "arrive": arrive,
                            "qty": q,
                            "cost_cny": cost,
                            "leg_picks": picks,
                        })
            continue

        # 5) 如果未来会缺货：必须补一票能赶得上的货（优先便宜的路线，但要能在缺货前到）
        #    我们用“可行路线集合”= lead <= (stockout_day - today).days
        need_by = (stockout_day - today).days
        candidate_routes = []
        for r in routes.values():
            if r.total_lead <= need_by:
                candidate_routes.append(r)

        # 如果海运赶不上，只能空运
        if not candidate_routes:
            candidate_routes = [routes["AIR"]]

        # 评估每条候选路线：在其到货日不超上限的前提下，选“单位成本最低”的那条
        best_option = None
        for r in candidate_routes:
            depart = today
            arrive = depart + timedelta(days=r.total_lead)
            if arrive > end_date:
                continue

            inv_before = jp_inventory_before(arrive)
            existing = jp_existing_arrival(arrive)
            q_max = math.floor(jp_cap - (inv_before + existing))
            if q_max < r.min_qty:
                continue

            # 这票的目标：尽量补到上限（防止频繁发票），但不能超过剩余发货目标/工厂库存
            q = min(q_max, remaining_to_ship, fac)
            q = int(math.floor(q))

            if q < r.min_qty:
                continue

            picks_cost = r.pick_cheapest_leg_carriers(q, fx)
            if picks_cost is None:
                continue
            picks, cost = picks_cost
            unit_cost = cost / q
            # tie-break：单位成本 -> 总成本 -> 票量大（减少票数）
            key = (unit_cost, cost, -q)
            if best_option is None or key < best_option["key"]:
                best_option = {
                    "key": key,
                    "route": r,
                    "depart": depart,
                    "arrive": arrive,
                    "qty": q,
                    "cost": cost,
                    "picks": picks,
                }

        if best_option is None:
            raise RuntimeError(f"无法在 {today} 修复未来缺货（可能是日本上限太紧/工厂库存不足/最小起运量过大）")

        # 执行最优补货票
        r = best_option["route"]
        q = best_option["qty"]
        fac -= q
        shipped_total += q
        arrivals[best_option["arrive"]] = arrivals.get(best_option["arrive"], 0) + q
        route_shipments.append({
            "route": r.name,
            "depart": best_option["depart"],
            "arrive": best_option["arrive"],
            "qty": q,
            "cost_cny": best_option["cost"],
            "leg_picks": best_option["picks"],
        })

    # 结束后做一次完整仿真验证（严检）
    jp = jp_init
    d = start_date
    stockout_days = 0
    overcap_days = 0
    jp_rows = []
    for i, dem in enumerate(demands):
        arr = arrivals.get(d, 0)
        before = jp
        after = jp + arr
        if after > jp_cap + 1e-9:
            overcap_days += 1
        end = after - dem
        if end < -1e-9:
            stockout_days += 1
        jp_rows.append([d, before, arr, after, dem, end])
        jp = end
        d += timedelta(days=1)

    jp_df = pd.DataFrame(jp_rows, columns=["日期","期初库存","到货","到货后","销量","期末库存"])

    summary = {
        "日本缺货天数": stockout_days,
        "日本超上限天数": overcap_days,
        "总发货量(到日本)": int(sum(arrivals.values())),
        "计划目标发货量": int(target_ship_total),
        "日本期末库存": float(jp_df["期末库存"].iloc[-1]),
        "日本最低期末库存": float(jp_df["期末库存"].min()),
        "日本最高到货后库存": float(jp_df["到货后"].max()),
        "总成本CNY(运输)": round(sum(x["cost_cny"] for x in route_shipments), 2)
    }
    return route_shipments, jp_df, summary


# =========================
# 4) 展开成“每一段运输”的系统录入行
# =========================
def expand_to_leg_plan(route_shipments, routes, fx):
    rows = []
    for s in route_shipments:
        r = routes[s["route"]]
        qty = s["qty"]
        depart = s["depart"]

        # 逐段推演：下一段起运=上一段到达日（默认当天衔接，不额外等待）
        current_depart = depart
        for (leg_name, _options), (_leg_name2, picked_carrier) in zip(r.legs, s["leg_picks"]):
            c = picked_carrier
            arrive = current_depart + timedelta(days=c.lead)
            rows.append({
                "路线": s["route"],
                "运输段": leg_name,
                "承运商": c.name,
                "单趟运量(件)": qty,
                "起运日期": current_depart.isoformat(),
                "承运趟数": 1,
                "多少天一趟": 0,  # 可后续再按同段起运日间隔填
                "到达日期": arrive.isoformat(),
                "运费预算(CNY)": round(c.cost_cny(qty, fx), 2),
            })
            current_depart = arrive

    df = pd.DataFrame(rows)
    # 给“多少天一趟”补上同段间隔
    df["起运日期_dt"] = pd.to_datetime(df["起运日期"])
    df.sort_values(["运输段","起运日期_dt"], inplace=True)
    df["多少天一趟"] = df.groupby("运输段")["起运日期_dt"].diff().dt.days.fillna(0).astype(int)
    df.drop(columns=["起运日期_dt"], inplace=True)
    return df


# =========================
# 5) 画图
# =========================
def plot_inventory(df, cap, title, filename):
    set_cn_font()
    fig, ax1 = plt.subplots(figsize=(12,5))
    ax1.plot(df["日期"], df["期末库存"], linewidth=2, label="期末库存")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axhline(cap, color="red", linestyle="--", linewidth=1.5, label=f"上限 {cap}")
    ax1.set_title(title)
    ax1.set_ylabel("库存（件）")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(df["日期"], df["到货"], alpha=0.25, label="到货量")
    ax2.set_ylabel("到货（件）")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    out = here(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[输出] {filename} -> {out}")


# =========================
# 6) 主程序（你只改这里也行）
# =========================
if __name__ == "__main__":
    # 汇率（你给的表）
    fx = {"CNY": 1.0, "USD": 7.1315, "JPY": 0.0656}

    # 两条路线的逐段承运商（按你截图）
    route_air = Route(
        name="AIR",
        legs=[
            ("宁波工厂->栎社机场", [
                Carrier("开源陆运", 1, 0.86, 2000, 2326, "CNY"),
                Carrier("易达快运", 1, 0.76, 6000, 7895, "CNY"),
                Carrier("城市配送", 1, 1.22,  500,  410, "CNY"),
            ]),
            ("栎社机场->东京机场", [
                Carrier("中国航空", 1, 3.62, 5000, 1382, "USD"),
            ]),
            ("东京机场->日本销售中心", [
                Carrier("国际陆运", 1, 25.74, 60000, 2332, "JPY"),
            ]),
        ],
        total_lead=3,
        min_qty=2332
    )

    route_sea = Route(
        name="SEA",
        legs=[
            ("宁波工厂->北仑码头", [
                Carrier("开源陆运", 1, 1.39, 2000, 1439, "CNY"),
                Carrier("易达快运", 1, 1.22, 6000, 4919, "CNY"),
                Carrier("城市配送", 1, 1.97,  500,  254, "CNY"),
            ]),
            ("北仑码头->东京港", [
                Carrier("中远海运", 9, 1.75, 5000, 2858, "USD"),
            ]),
            ("东京港->日本销售中心", [
                Carrier("国际陆运", 1, 45.24, 60000, 1327, "JPY"),
            ]),
        ],
        total_lead=11,
        min_qty=2858
    )

    routes = {"AIR": route_air, "SEA": route_sea}

    # 运行排程（只满足日本）
    route_shipments, jp_df, summary = plan_japan_only(
        start_date=date(2025,1,1),
        demand_total=51028,
        demand_daily=851,
        jp_init=7000,
        jp_cap=8000,
        fac_init=5000,
        fac_cap=6000,
        fac_prod_daily=1861,
        fac_prod_end=date(2025,2,26),
        routes=routes,
        fx=fx
    )

    print("===== 日本优先：排程结果摘要 =====")
    print(summary)

    # 展开成“每段运输计划”（系统录入用）
    leg_df = expand_to_leg_plan(route_shipments, routes, fx)

    # 输出文件
    jp_df.to_csv(here("jp_daily.csv"), index=False, encoding="utf-8-sig")
    leg_df.to_csv(here("jp_leg_plan.csv"), index=False, encoding="utf-8-sig")

    print(f"[输出] jp_daily.csv -> {here('jp_daily.csv')}")
    print(f"[输出] jp_leg_plan.csv -> {here('jp_leg_plan.csv')}")

    # 画日本库存图
    plot_inventory(jp_df, cap=8000, title="日本销售中心库存曲线（优先满足日本）", filename="jp_inventory.png")
