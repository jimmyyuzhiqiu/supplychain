
# -*- coding: utf-8 -*-
import os
import math
from dataclasses import dataclass
from datetime import date, timedelta

import matplotlib
matplotlib.use("Agg")  # ✅ 强制落盘 PNG（不依赖GUI）

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 0) 工具：路径 & 中文字体
# =========================
def here(fname: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)

def set_cn_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1) 数据结构：承运商/路线
# =========================
@dataclass
class Carrier:
    name: str
    lead: int               # 天
    unit_cost: float        # 单价（元/件、或美元/件、或日元/件）
    start_fee: float        # 起步价（同币种）
    min_qty: int            # 最小起运量（件）
    currency: str = "CNY"   # CNY / USD / JPY

    def cost_cny(self, qty: int, fx: dict) -> float:
        """费用 = max(起步价, 单价*qty)，并按汇率折CNY"""
        if qty <= 0:
            return 0.0
        raw = max(self.start_fee, self.unit_cost * qty)
        return raw * fx[self.currency]

@dataclass
class Route:
    name: str
    legs: list              # [(leg_name, [Carrier,...]), ...]
    total_lead: int         # 总提前期（天）
    min_qty: int            # 路线整票最小起运量（由各段min约束共同决定）

    def pick_cheapest_leg_carriers(self, qty: int, fx: dict):
        """给定 qty，逐段选择满足 min_qty 的最便宜承运商"""
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
# 2) 需求序列（保证总量准确）
# =========================
def build_daily_demand(total: int, daily_avg: int):
    days = math.ceil(total / daily_avg)
    dem = [daily_avg] * days
    dem[-1] = total - daily_avg * (days - 1)
    return dem


# =========================
# 3) 仿真：工厂+日本（关键：违规时也返回DF，不返回None）
#    顺序：工厂生产 -> 工厂起运 -> 日本到货 -> 日本消耗
# =========================
def simulate_factory_and_jp(
    start_date: date,
    demands: list,
    jp_init: int,
    jp_cap: int,
    fac_init: int,
    fac_cap: int,
    fac_prod_daily: int,
    fac_prod_end: date,
    shipments: list,   # [{'route':Route,'depart':date,'qty':int,'arrive':date(optional)}]
):
    horizon = len(demands)
    end_date = start_date + timedelta(days=horizon - 1)

    # 聚合：按起运日
    ship_by_depart = {}
    # 聚合：按到货日（日本）
    arrivals = {}

    for s in shipments:
        ship_by_depart.setdefault(s["depart"], []).append(s)
        arr = s["depart"] + timedelta(days=s["route"].total_lead)
        if arr <= end_date:
            arrivals[arr] = arrivals.get(arr, 0) + int(s["qty"])

    fac = float(fac_init)
    jp = float(jp_init)

    fac_rows = []
    jp_rows = []

    for t in range(horizon):
        d = start_date + timedelta(days=t)

        # --- 工厂生产 ---
        fac_start = fac
        prod = fac_prod_daily if d <= fac_prod_end else 0
        if prod > 0:
            fac = min(fac_cap, fac + prod)

        # --- 工厂起运 ---
        dep_list = ship_by_depart.get(d, [])
        dep_qty = sum(int(x["qty"]) for x in dep_list)
        fac_end = fac - dep_qty

        fac_rows.append([d, fac_start, prod, dep_qty, fac_end])

        if fac_end < -1e-9:
            fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存"])
            jp_df  = pd.DataFrame(jp_rows,  columns=["日期","期初库存","到货","到货后","销量","期末库存"])
            # a=发货前库存(含生产后), b=当日总发货
            return fac_df, jp_df, ("FACT_NEG", d, fac, dep_qty)

        fac = fac_end

        # --- 日本到货 ---
        jp_start = jp
        arr_qty = arrivals.get(d, 0)
        jp_after = jp + arr_qty

        # 先记录行，期末稍后补
        jp_rows.append([d, jp_start, arr_qty, jp_after, demands[t], None])

        if jp_after > jp_cap + 1e-9:
            fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存"])
            jp_df  = pd.DataFrame(jp_rows,  columns=["日期","期初库存","到货","到货后","销量","期末库存"])
            return fac_df, jp_df, ("JP_OVER", d, jp_after, jp_cap)

        # --- 日本消耗 ---
        jp_end = jp_after - demands[t]
        jp_rows[-1][-1] = jp_end

        if jp_end < -1e-9:
            fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存"])
            jp_df  = pd.DataFrame(jp_rows,  columns=["日期","期初库存","到货","到货后","销量","期末库存"])
            # a=到货后库存, b=当日需求
            return fac_df, jp_df, ("JP_STOCKOUT", d, jp_after, demands[t])

        jp = jp_end

    fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存"])
    jp_df  = pd.DataFrame(jp_rows,  columns=["日期","期初库存","到货","到货后","销量","期末库存"])
    return fac_df, jp_df, None


# =========================
# 4) 计划器（稳定版）：迭代修复缺货/超限/工厂负库存
#    目标：先保证“日本不缺货且不超上限”，并尽量多走海运（单位成本更低）
# =========================
def plan_japan_only_stable(
    start_date: date,
    demand_total: int,
    demand_daily: int,
    jp_init: int,
    jp_cap: int,
    fac_init: int,
    fac_cap: int,
    fac_prod_daily: int,
    fac_prod_end: date,
    routes: dict,   # {"SEA":Route,"AIR":Route}
    fx: dict,
    max_iters: int = 600,
    search_back_days: int = 20,  # 修复缺货时，允许起运日向前回溯多少天
):
    demands = build_daily_demand(demand_total, demand_daily)
    horizon = len(demands)
    end_date = start_date + timedelta(days=horizon - 1)

    shipments = []  # 每票：{"route":Route,"depart":date,"qty":int,"arrive":date,"leg_picks":...,"cost_cny":...}

    def recalc_ticket(s):
        """重算该票的到货日、段内承运商选择、成本"""
        s["arrive"] = s["depart"] + timedelta(days=s["route"].total_lead)
        picks_cost = s["route"].pick_cheapest_leg_carriers(int(s["qty"]), fx)
        if picks_cost is None:
            return False
        s["leg_picks"], s["cost_cny"] = picks_cost
        return True

    def total_cost():
        return round(sum(float(s.get("cost_cny", 0.0)) for s in shipments), 2)

    # 主循环：仿真->发现第一个违规->修复
    for it in range(1, max_iters + 1):
        fac_df, jp_df, viol = simulate_factory_and_jp(
            start_date, demands,
            jp_init, jp_cap,
            fac_init, fac_cap, fac_prod_daily, fac_prod_end,
            shipments
        )

        if viol is None:
            # ✅ 可行解
            summary = {
                "迭代次数": it,
                "票数": len(shipments),
                "总成本CNY(运输)": total_cost(),
                "日本期末库存": float(jp_df["期末库存"].iloc[-1]),
                "日本最低期末库存": float(jp_df["期末库存"].min()),
                "日本最高到货后库存": float(jp_df["到货后"].max()),
                "工厂期末库存": float(fac_df["期末库存"].iloc[-1]),
                "工厂最低期末库存": float(fac_df["期末库存"].min()),
            }
            # 排序一下输出更稳
            shipments_sorted = sorted(shipments, key=lambda x: (x["depart"], x["route"].name))
            return shipments_sorted, fac_df, jp_df, summary

        vtype, vday, a, b = viol

        # -------------------------
        # 4.1 日本超上限：削减/删除当天到货的最后一票
        # -------------------------
        if vtype == "JP_OVER":
            overflow = float(a) - float(b)  # a=到货后库存，b=cap
            # 找到所有 arrive=vday 的票
            candidates = [s for s in shipments if s.get("arrive") == vday]
            if not candidates:
                raise RuntimeError(f"[JP_OVER] {vday} 超上限，但没有任何票到货于该日，无法回退。")

            # 选择“起运最晚”的那票先削（最符合回溯逻辑）
            s = max(candidates, key=lambda x: x["depart"])
            new_qty = int(max(0, math.floor(int(s["qty"]) - overflow)))

            if new_qty < s["route"].min_qty:
                shipments.remove(s)
            else:
                s["qty"] = new_qty
                if not recalc_ticket(s):
                    shipments.remove(s)

            continue

        # -------------------------
        # 4.2 工厂负库存：削减/删除当天起运的最大一票
        # -------------------------
        if vtype == "FACT_NEG":
            # a=发货前库存(含生产后)，b=当天总发货
            available = float(a)
            dep_total = float(b)
            deficit = dep_total - available

            cand = [s for s in shipments if s["depart"] == vday]
            if not cand:
                raise RuntimeError(f"[FACT_NEG] {vday} 工厂负库存，但没有票在该日起运，无法回退。")

            s = max(cand, key=lambda x: int(x["qty"]))
            new_qty = int(max(0, math.floor(int(s["qty"]) - deficit)))

            if new_qty < s["route"].min_qty:
                shipments.remove(s)
            else:
                s["qty"] = new_qty
                if not recalc_ticket(s):
                    shipments.remove(s)

            continue

        # -------------------------
        # 4.3 日本缺货：插入一票“能赶得上且不超上限且工厂凑得出最小量”的运输
        #     海运优先（单位成本更低），赶不上才空运
        # -------------------------
        if vtype != "JP_STOCKOUT":
            raise RuntimeError(f"未知违规类型：{viol}")

        # vday 缺货：我们尝试安排一票到货日 <= vday（通常到货日= vday 最好）
        best_insert = None

        # 先按成本倾向：SEA -> AIR
        for r in [routes["SEA"], routes["AIR"]]:
            latest_depart = vday - timedelta(days=r.total_lead)
            if latest_depart < start_date:
                continue

            # 从 latest_depart 往前回溯 search_back_days 天找可行起运日
            for k in range(0, search_back_days + 1):
                dep = latest_depart - timedelta(days=k)
                if dep < start_date:
                    break

                arrive = dep + timedelta(days=r.total_lead)
                if arrive > end_date or arrive > vday:
                    continue

                # 头寸：日本到货日“到货前库存”和“当天已排到货”
                jp_row = jp_df[jp_df["日期"] == arrive]
                if jp_row.empty:
                    continue

                jp_before = float(jp_row["期初库存"].iloc[0])
                existing_arrival = float(jp_row["到货"].iloc[0])
                headroom = math.floor(jp_cap - (jp_before + existing_arrival))
                if headroom < r.min_qty:
                    continue

                # 工厂 dep 当天可用于新增发货的量：
                # fac_df 的“期末库存”是当天已发货后的；要回推发货前可用=期末+当日发货
                fac_row = fac_df[fac_df["日期"] == dep]
                if fac_row.empty:
                    continue
                fac_available = float(fac_row["期末库存"].iloc[0]) + float(fac_row["发货"].iloc[0])
                qmax = int(min(headroom, math.floor(fac_available)))
                if qmax < r.min_qty:
                    continue

                # 票量：尽量发满（减少票数/起步价次数），但不超过 qmax
                q = int(qmax)

                picks_cost = r.pick_cheapest_leg_carriers(q, fx)
                if picks_cost is None:
                    continue
                picks, cost = picks_cost

                # 评分：单位成本优先 -> 总成本 -> 起运越晚越好（减少提前占用）
                key = (cost / q, cost, -dep.toordinal())
                if best_insert is None or key < best_insert["key"]:
                    best_insert = {
                        "key": key,
                        "route": r,
                        "depart": dep,
                        "qty": q,
                        "arrive": arrive,
                        "leg_picks": picks,
                        "cost_cny": cost,
                    }

        if best_insert is None:
            raise RuntimeError(
                f"[JP_STOCKOUT] 无法修复缺货：缺货日={vday}。"
                f"常见原因：日本上限头寸 < 最小起运量，或工厂在可起运窗口内凑不出最小起运量。"
            )

        shipments.append(best_insert)
        continue

    raise RuntimeError("超过最大迭代次数仍未找到可行方案，请扩大 search_back_days 或放宽约束。")


# =========================
# 5) 输出：展开成系统录入（按运输段）
#    规则：默认允许“到达当日即可发出下一段”（cross-dock，无额外等待）
# =========================
def expand_to_leg_plan(shipments, fx):
    rows = []
    for s in shipments:
        r = s["route"]
        qty = int(s["qty"])
        current_depart = s["depart"]

        # leg_picks: [(leg_name, Carrier), ...]
        for leg_name, picked in s["leg_picks"]:
            arrive = current_depart + timedelta(days=picked.lead)
            rows.append({
                "路线": r.name,
                "运输段": leg_name,
                "承运商": picked.name,
                "单趟运量(件)": qty,
                "起运日期": current_depart.isoformat(),
                "承运趟数": 1,
                "多少天一趟": 0,  # 后面再按同段间隔填
                "到达日期": arrive.isoformat(),
                "运费预算(CNY)": round(picked.cost_cny(qty, fx), 2),
            })
            # 下一段起运日=上一段到达日（cross-dock）
            current_depart = arrive

    df = pd.DataFrame(rows)
    # 补“多少天一趟”：按运输段分组，看起运日期差
    df["起运日期_dt"] = pd.to_datetime(df["起运日期"])
    df.sort_values(["运输段","起运日期_dt"], inplace=True)
    df["多少天一趟"] = df.groupby("运输段")["起运日期_dt"].diff().dt.days.fillna(0).astype(int)
    df.drop(columns=["起运日期_dt"], inplace=True)
    return df


def build_route_plan_summary(shipments):
    rows = []
    for s in shipments:
        rows.append({
            "路线": s["route"].name,
            "起运日期": s["depart"].isoformat(),
            "到达日期(日本)": s["arrive"].isoformat(),
            "单票量(件)": int(s["qty"]),
            "成本(CNY)": round(float(s["cost_cny"]), 2),
        })
    df = pd.DataFrame(rows).sort_values(["到达日期(日本)","路线","起运日期"])
    return df


# =========================
# 6) 画图
# =========================
def plot_inventory(df, cap, title, filename, bar_col):
    set_cn_font()
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["日期"], df["期末库存"], linewidth=2, label="期末库存")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.axhline(cap, color="red", linestyle="--", linewidth=1.5, label=f"上限 {cap}")
    ax1.set_title(title)
    ax1.set_ylabel("库存（件）")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(df["日期"], df[bar_col], alpha=0.25, label=bar_col)
    ax2.set_ylabel(bar_col + "（件）")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper right")

    out = here(filename)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"[输出] {filename} -> {out}")


# =========================
# 7) 主程序：只满足日本
# =========================
if __name__ == "__main__":
    # ---- 汇率（你给的表）
    fx = {"CNY": 1.0, "USD": 7.1315, "JPY": 0.0656}

    # ---- AIR 路线：宁波工厂->栎社机场->东京机场->日本销售中心（总lead=3）
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

    # ---- SEA 路线：宁波工厂->北仑码头->东京港->日本销售中心（总lead=11）
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

    # ---- 输入参数（按你提供）
    start_date = date(2025, 1, 1)

    jp_total = 51028
    jp_daily = 851
    jp_init = 7000
    jp_cap  = 8000

    fac_init = 5000
    fac_cap  = 6000
    fac_prod_daily = 1861
    fac_prod_end   = date(2025, 2, 26)

    # ---- 计算日本优先计划
    shipments, fac_df, jp_df, summary = plan_japan_only_stable(
        start_date=start_date,
        demand_total=jp_total,
        demand_daily=jp_daily,
        jp_init=jp_init,
        jp_cap=jp_cap,
        fac_init=fac_init,
        fac_cap=fac_cap,
        fac_prod_daily=fac_prod_daily,
        fac_prod_end=fac_prod_end,
        routes=routes,
        fx=fx,
        max_iters=600,
        search_back_days=20
    )

    print("\n===== 日本优先：可行方案摘要 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # ---- 输出 CSV
    leg_df = expand_to_leg_plan(shipments, fx)
    route_df = build_route_plan_summary(shipments)

    fac_csv = here("factory_daily.csv")
    jp_csv  = here("jp_daily.csv")
    leg_csv = here("jp_leg_plan.csv")
    route_csv = here("jp_route_plan.csv")

    fac_df.to_csv(fac_csv, index=False, encoding="utf-8-sig")
    jp_df.to_csv(jp_csv,  index=False, encoding="utf-8-sig")
    leg_df.to_csv(leg_csv, index=False, encoding="utf-8-sig")
    route_df.to_csv(route_csv, index=False, encoding="utf-8-sig")

    print(f"\n[输出] factory_daily.csv -> {fac_csv}")
    print(f"[输出] jp_daily.csv      -> {jp_csv}")
    print(f"[输出] jp_leg_plan.csv   -> {leg_csv}")
    print(f"[输出] jp_route_plan.csv -> {route_csv}")

    # ---- 输出 PNG
    plot_inventory(jp_df, cap=jp_cap, title="日本销售中心库存曲线（优先满足日本）", filename="jp_inventory.png", bar_col="到货")
    plot_inventory(fac_df, cap=fac_cap, title="宁波工厂库存曲线（日本优先发货）", filename="factory_inventory.png", bar_col="发货")

    print("\n==== 完成：文件已输出到脚本所在目录 ====")
