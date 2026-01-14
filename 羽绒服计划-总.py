
# -*- coding: utf-8 -*-
import os
import math
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt


def here(fname: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)

def set_cn_font():
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC",
        "WenQuanYi Zen Hei", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Carrier:
    name: str
    lead: int
    unit_cost: float
    start_fee: float
    min_qty: int
    currency: str = "CNY"

    def cost_cny(self, qty: int, fx: dict) -> float:
        if qty <= 0:
            return 0.0
        raw = max(self.start_fee, self.unit_cost * qty)
        return raw * fx[self.currency]


@dataclass
class RouteTemplate:
    name: str
    dest: str
    legs: list  # [(leg_name, [Carrier...]), ...]

    def best_combo(self, qty: int, fx: dict, max_lead: int | None = None):
        if qty <= 0:
            return None

        options_per_leg = []
        for leg_name, carriers in self.legs:
            feas = [c for c in carriers if qty >= c.min_qty]
            if not feas:
                return None
            options_per_leg.append([(leg_name, c) for c in feas])

        best = None
        for combo in product(*options_per_leg):
            total_lead = sum(c.lead for _, c in combo)
            if max_lead is not None and total_lead > max_lead:
                continue
            total_cost = sum(c.cost_cny(qty, fx) for _, c in combo)
            key = (total_cost, total_lead)
            if best is None or key < best["key"]:
                best = {"key": key, "picks": list(combo), "cost": total_cost, "lead": total_lead}
        if best is None:
            return None
        return best["picks"], best["cost"], best["lead"]


@dataclass
class Node:
    name: str
    init: int
    cap: int
    total_demand: int
    daily_avg: int


@dataclass
class Factory:
    init: int
    cap: int
    prod_daily: int
    prod_end: date
    allow_overcap: bool = True
    overcap_fee: float = 5.0


def build_daily_demand(total: int, daily_avg: int):
    days = math.ceil(total / daily_avg)
    dem = [daily_avg] * days
    dem[-1] = total - daily_avg * (days - 1)
    return dem

def pad_demands(dem: list, horizon: int):
    if len(dem) >= horizon:
        return dem[:horizon]
    return dem + [0] * (horizon - len(dem))


def simulate_system(start_date, horizon, factory, demands_by_node, nodes, shipments, fx):
    end_date = start_date + timedelta(days=horizon - 1)

    ship_by_depart = {}
    arrivals = {}
    transport_cost = 0.0

    for s in shipments:
        ship_by_depart.setdefault(s["depart"], []).append(s)
        if s["arrive"] <= end_date:
            key = (s["dest"], s["arrive"])
            arrivals[key] = arrivals.get(key, 0) + int(s["qty"])
        transport_cost += float(s["cost_cny"])

    fac = float(factory.init)
    inv = {n: float(nodes[n].init) for n in nodes.keys()}

    fac_rows = []
    node_rows = {n: [] for n in nodes.keys()}
    factory_overcap_penalty = 0.0

    for t in range(horizon):
        d = start_date + timedelta(days=t)

        fac_start = fac
        prod = factory.prod_daily if d <= factory.prod_end else 0
        fac += prod

        dep_list = ship_by_depart.get(d, [])
        dep_qty = sum(int(x["qty"]) for x in dep_list)
        fac_end = fac - dep_qty

        fac_rows.append([d, fac_start, prod, dep_qty, fac_end, 0.0, 0.0])

        if fac_end < -1e-9:
            fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存","超限量","当日超限费"])
            node_dfs = {k: pd.DataFrame(node_rows[k], columns=["日期","期初库存","到货","到货后","销量","期末库存"]) for k in nodes.keys()}
            costs = {"transport_cost": transport_cost, "factory_overcap_penalty": factory_overcap_penalty, "total_cost": transport_cost + factory_overcap_penalty}
            return fac_df, node_dfs, ("FACT_NEG", d, fac, dep_qty), costs

        over_qty = max(0.0, fac_end - factory.cap)
        if over_qty > 0:
            day_pen = over_qty * factory.overcap_fee
            factory_overcap_penalty += day_pen
            fac_rows[-1][-2] = over_qty
            fac_rows[-1][-1] = day_pen
            if not factory.allow_overcap:
                fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存","超限量","当日超限费"])
                node_dfs = {k: pd.DataFrame(node_rows[k], columns=["日期","期初库存","到货","到货后","销量","期末库存"]) for k in nodes.keys()}
                costs = {"transport_cost": transport_cost, "factory_overcap_penalty": factory_overcap_penalty, "total_cost": transport_cost + factory_overcap_penalty}
                return fac_df, node_dfs, ("FACT_OVER", d, fac_end, factory.cap), costs

        fac = fac_end

        for n in nodes.keys():
            node = nodes[n]
            dem = demands_by_node[n][t]
            start_inv = inv[n]
            arr_qty = arrivals.get((n, d), 0)
            after_arr = start_inv + arr_qty

            node_rows[n].append([d, start_inv, arr_qty, after_arr, dem, None])

            if after_arr > node.cap + 1e-9:
                fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存","超限量","当日超限费"])
                node_dfs = {k: pd.DataFrame(node_rows[k], columns=["日期","期初库存","到货","到货后","销量","期末库存"]) for k in nodes.keys()}
                costs = {"transport_cost": transport_cost, "factory_overcap_penalty": factory_overcap_penalty, "total_cost": transport_cost + factory_overcap_penalty}
                return fac_df, node_dfs, ("NODE_OVER", n, d, after_arr, node.cap), costs

            end_inv = after_arr - dem
            node_rows[n][-1][-1] = end_inv

            if end_inv < -1e-9:
                fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存","超限量","当日超限费"])
                node_dfs = {k: pd.DataFrame(node_rows[k], columns=["日期","期初库存","到货","到货后","销量","期末库存"]) for k in nodes.keys()}
                costs = {"transport_cost": transport_cost, "factory_overcap_penalty": factory_overcap_penalty, "total_cost": transport_cost + factory_overcap_penalty}
                return fac_df, node_dfs, ("NODE_STOCKOUT", n, d, after_arr, dem), costs

            inv[n] = end_inv

    fac_df = pd.DataFrame(fac_rows, columns=["日期","期初库存","产量","发货","期末库存","超限量","当日超限费"])
    node_dfs = {n: pd.DataFrame(node_rows[n], columns=["日期","期初库存","到货","到货后","销量","期末库存"]) for n in nodes.keys()}
    costs = {"transport_cost": transport_cost, "factory_overcap_penalty": factory_overcap_penalty, "total_cost": transport_cost + factory_overcap_penalty}
    return fac_df, node_dfs, None, costs


def plan_all_nodes_with_factory_overcap_tradeoff_v12(
    start_date: date,
    nodes: dict,
    factory: Factory,
    route_templates: list,
    fx: dict,
    max_repair_iters: int = 2500,
    max_improve_iters: int = 700,
    search_back_days: int = 25,
):
    demands_raw = {n: build_daily_demand(nodes[n].total_demand, nodes[n].daily_avg) for n in nodes.keys()}
    horizon = max(len(x) for x in demands_raw.values())
    demands_by_node = {n: pad_demands(demands_raw[n], horizon) for n in nodes.keys()}
    end_date = start_date + timedelta(days=horizon - 1)

    need_total = {n: max(0, nodes[n].total_demand - nodes[n].init) for n in nodes.keys()}

    rt_by_dest = {}
    for rt in route_templates:
        rt_by_dest.setdefault(rt.dest, []).append(rt)

    shipments = []

    def shipped_to_node(n):
        return sum(int(s["qty"]) for s in shipments if s["dest"] == n)

    def objective(costs):
        return float(costs["total_cost"])

    # ========== A) Repair ==========
    for _ in range(max_repair_iters):
        fac_df, node_dfs, viol, costs = simulate_system(
            start_date, horizon, factory, demands_by_node, nodes, shipments, fx
        )
        if viol is None:
            break

        if viol[0] == "NODE_STOCKOUT":
            _, n, vday, after_arr, dem = viol

            remain = need_total[n] - shipped_to_node(n)
            if remain < 0:
                remain = 0

            best_insert = None

            for rt in rt_by_dest[n]:
                for back in range(0, search_back_days + 1):
                    dep = vday - timedelta(days=back)
                    if dep < start_date:
                        break

                    fac_row = fac_df[fac_df["日期"] == dep]
                    if fac_row.empty:
                        continue

                    # ✅ v1.2：新增票可用量=当前计划下 dep 当天“剩余可发量”（期末库存）
                    fac_spare = max(0.0, float(fac_row["期末库存"].iloc[0]))
                    if fac_spare <= 0:
                        continue

                    max_lead = (vday - dep).days
                    if max_lead < 0:
                        continue

                    for lead_days in range(0, max_lead + 1):
                        arrive = dep + timedelta(days=lead_days)
                        if arrive > vday or arrive > end_date:
                            continue

                        row = node_dfs[n][node_dfs[n]["日期"] == arrive]
                        if row.empty:
                            continue

                        headroom = math.floor(nodes[n].cap - (float(row["期初库存"].iloc[0]) + float(row["到货"].iloc[0])))
                        if headroom <= 0:
                            continue

                        # ✅ v1.2：允许“为满足 min_qty 的超发缓冲”
                        # 票量上限：不能超过头寸/工厂剩余；同时不让超发太多：最多 remain + buffer(min_qty)
                        # 若 remain=0（纯时点问题），也允许发 min_qty（只要 cap 允许）
                        buffer_cap = 0  # 每个 rt 的 min_qty 可能不同 -> 我们用 combo 可行性来筛
                        qty_upper = int(min(headroom, math.floor(fac_spare)))

                        if qty_upper <= 0:
                            continue

                        # 先尝试“尽量发满但不太夸张”：min(qty_upper, remain + qty_upper 的下限约束会由 min_qty 决定)
                        # 我们把“超发上限”设为 remain + qty_upper 的最小可行值：remain + 5000 不现实
                        # 这里采用 remain + 3000（足够覆盖 JP 的 2332/2858），同时仍会被 headroom 限制
                        overship_limit = int(remain + 3000)
                        qty_candidate = int(min(qty_upper, overship_limit))

                        if qty_candidate <= 0:
                            continue

                        combo = rt.best_combo(qty_candidate, fx, max_lead=lead_days)
                        if combo is None:
                            # 若 qty_candidate 不可行（通常因为 < min_qty），再尝试把 qty 直接抬到 headroom/工厂允许范围内的更大值
                            combo = rt.best_combo(qty_upper, fx, max_lead=lead_days)
                            if combo is None:
                                continue
                            qty_candidate = qty_upper

                        picks, cost, real_lead = combo
                        real_arrive = dep + timedelta(days=real_lead)
                        if real_arrive != arrive:
                            continue

                        key = (cost / qty_candidate, cost, -dep.toordinal())
                        if best_insert is None or key < best_insert["key"]:
                            best_insert = {
                                "key": key,
                                "dest": n,
                                "route_name": rt.name,
                                "depart": dep,
                                "qty": int(qty_candidate),
                                "arrive": real_arrive,
                                "picks": picks,
                                "cost_cny": float(cost),
                            }

            if best_insert is None:
                # 输出诊断：当日 remain、可能头寸、以及工厂 spare 的极值
                print("\n[诊断] 缺货无法修复：")
                print("  节点:", n, "缺货日:", vday, "remain(理论还需发):", need_total[n] - shipped_to_node(n))
                # 工厂在回溯窗口内最大 spare
                d0 = vday - timedelta(days=search_back_days)
                fac_window = fac_df[(fac_df["日期"] >= d0) & (fac_df["日期"] <= vday)]
                if not fac_window.empty:
                    print("  工厂 spare(期末库存) 最大值:", float(fac_window["期末库存"].max()))
                raise RuntimeError(f"[缺货无法修复] 节点={n}, 缺货日={vday}。很可能是 remain 太小 < min_qty，或头寸/工厂剩余不足。")

            shipments.append(best_insert)
            continue

        if viol[0] == "NODE_OVER":
            _, n, vday, after_arr, cap = viol
            overflow = float(after_arr) - float(cap)

            candidates = [s for s in shipments if s["dest"] == n and s["arrive"] == vday]
            if not candidates:
                raise RuntimeError(f"[超限无法回退] 节点={n} 日期={vday} 没有票到货。")

            s = max(candidates, key=lambda x: x["depart"])
            new_qty = int(max(0, math.floor(int(s["qty"]) - overflow)))

            if new_qty <= 0:
                shipments.remove(s)
            else:
                rt = next(rt for rt in route_templates if rt.name == s["route_name"] and rt.dest == s["dest"])
                combo = rt.best_combo(new_qty, fx, None)
                if combo is None:
                    shipments.remove(s)
                else:
                    picks, cost, lead = combo
                    s["qty"] = new_qty
                    s["picks"] = picks
                    s["cost_cny"] = float(cost)
                    s["arrive"] = s["depart"] + timedelta(days=lead)
            continue

        if viol[0] == "FACT_NEG":
            _, vday, fac_before_ship, dep_total = viol
            deficit = float(dep_total) - float(fac_before_ship)

            candidates = [s for s in shipments if s["depart"] == vday]
            if not candidates:
                raise RuntimeError(f"[工厂负库存无法回退] {vday} 无票起运。")

            s = max(candidates, key=lambda x: int(x["qty"]))
            new_qty = int(max(0, math.floor(int(s["qty"]) - deficit)))

            if new_qty <= 0:
                shipments.remove(s)
            else:
                rt = next(rt for rt in route_templates if rt.name == s["route_name"] and rt.dest == s["dest"])
                combo = rt.best_combo(new_qty, fx, None)
                if combo is None:
                    shipments.remove(s)
                else:
                    picks, cost, lead = combo
                    s["qty"] = new_qty
                    s["picks"] = picks
                    s["cost_cny"] = float(cost)
                    s["arrive"] = s["depart"] + timedelta(days=lead)
            continue

        if viol[0] == "FACT_OVER":
            raise RuntimeError(f"工厂超限硬约束触发：{viol}")

        raise RuntimeError(f"未知违规：{viol}")

    # Repair 后必须可行
    fac_df, node_dfs, viol, costs = simulate_system(
        start_date, horizon, factory, demands_by_node, nodes, shipments, fx
    )
    if viol is not None:
        raise RuntimeError(f"修复阶段结束但仍不可行：{viol}")

    # ========== B) Improve（严格不超发：不再使用 overship_limit）==========
    for _ in range(max_improve_iters):
        fac_df, node_dfs, viol, costs_before = simulate_system(
            start_date, horizon, factory, demands_by_node, nodes, shipments, fx
        )
        if viol is not None:
            raise RuntimeError(f"改进阶段出现不可行：{viol}")

        base_obj = objective(costs_before)

        fac_df["超限量_tmp"] = (fac_df["期末库存"] - factory.cap).clip(lower=0)
        top_days = fac_df.sort_values("超限量_tmp", ascending=False).head(8)
        if top_days["超限量_tmp"].max() <= 1e-9:
            break

        best_move = None

        for _, r in top_days.iterrows():
            dep_day = r["日期"]
            if float(r["超限量_tmp"]) <= 0:
                continue

            fac_row = fac_df[fac_df["日期"] == dep_day]
            fac_spare = max(0.0, float(fac_row["期末库存"].iloc[0]))
            if fac_spare <= 0:
                continue

            for n in nodes.keys():
                remain = need_total[n] - shipped_to_node(n)
                if remain <= 0:
                    continue

                for rt in rt_by_dest[n]:
                    for lead_days in range(0, 31):
                        arrive = dep_day + timedelta(days=lead_days)
                        row = node_dfs[n][node_dfs[n]["日期"] == arrive]
                        if row.empty:
                            continue

                        headroom = math.floor(nodes[n].cap - (float(row["期初库存"].iloc[0]) + float(row["到货"].iloc[0])))
                        if headroom <= 0:
                            continue

                        qty_max = int(min(headroom, math.floor(fac_spare), remain))
                        if qty_max <= 0:
                            continue

                        combo = rt.best_combo(qty_max, fx, max_lead=lead_days)
                        if combo is None:
                            continue
                        picks, cost, real_lead = combo
                        real_arrive = dep_day + timedelta(days=real_lead)

                        candidate = {
                            "dest": n,
                            "route_name": rt.name,
                            "depart": dep_day,
                            "qty": int(qty_max),
                            "arrive": real_arrive,
                            "picks": picks,
                            "cost_cny": float(cost),
                        }

                        _, _, viol2, costs_after = simulate_system(
                            start_date, horizon, factory, demands_by_node, nodes, shipments + [candidate], fx
                        )
                        if viol2 is not None:
                            continue

                        new_obj = objective(costs_after)
                        improvement = base_obj - new_obj
                        if improvement > 1e-6:
                            key = (new_obj, -improvement)
                            if best_move is None or key < best_move["key"]:
                                best_move = {"key": key, "candidate": candidate}

        if best_move is None:
            break
        shipments.append(best_move["candidate"])

    fac_df, node_dfs, viol, final_costs = simulate_system(
        start_date, horizon, factory, demands_by_node, nodes, shipments, fx
    )
    if viol is not None:
        raise RuntimeError(f"最终方案不可行：{viol}")

    summary = {
        "总票数": len(shipments),
        "运输成本CNY": round(final_costs["transport_cost"], 2),
        "工厂超限费CNY": round(final_costs["factory_overcap_penalty"], 2),
        "总成本CNY(运输+超限费)": round(final_costs["total_cost"], 2),
        "工厂最大超限量": float((fac_df["期末库存"] - factory.cap).clip(lower=0).max()),
    }
    for n in nodes.keys():
        df = node_dfs[n]
        summary[f"{n}_最低期末库存"] = float(df["期末库存"].min())
        summary[f"{n}_最高到货后库存"] = float(df["到货后"].max())
        summary[f"{n}_期末库存"] = float(df["期末库存"].iloc[-1])

    return shipments, fac_df, node_dfs, summary


def route_plan_df(shipments):
    rows = []
    for s in shipments:
        rows.append({
            "目的地": s["dest"],
            "路线": s["route_name"],
            "起运日期": s["depart"].isoformat(),
            "到达日期": s["arrive"].isoformat(),
            "单票量(件)": int(s["qty"]),
            "成本(CNY)": round(float(s["cost_cny"]), 2),
        })
    return pd.DataFrame(rows).sort_values(["到达日期","目的地","路线","起运日期"])

def leg_plan_df(shipments, fx):
    rows = []
    for s in shipments:
        qty = int(s["qty"])
        cur_depart = s["depart"]
        for leg_name, carrier in s["picks"]:
            arrive = cur_depart + timedelta(days=carrier.lead)
            rows.append({
                "目的地": s["dest"],
                "路线": s["route_name"],
                "运输段": leg_name,
                "承运商": carrier.name,
                "单趟运量(件)": qty,
                "起运日期": cur_depart.isoformat(),
                "承运趟数": 1,
                "多少天一趟": 0,
                "到达日期": arrive.isoformat(),
                "运费预算(CNY)": round(carrier.cost_cny(qty, fx), 2),
            })
            cur_depart = arrive

    df = pd.DataFrame(rows)
    df["起运日期_dt"] = pd.to_datetime(df["起运日期"])
    df.sort_values(["运输段","起运日期_dt"], inplace=True)
    df["多少天一趟"] = df.groupby("运输段")["起运日期_dt"].diff().dt.days.fillna(0).astype(int)
    df.drop(columns=["起运日期_dt"], inplace=True)
    return df


def plot_node(df, cap, title, filename, bar_col):
    set_cn_font()
    fig, ax1 = plt.subplots(figsize=(12,5))
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


if __name__ == "__main__":
    fx = {"CNY": 1.0, "USD": 7.1315, "JPY": 0.0656}

    nodes = {
        "JP":    Node("JP", init=7000, cap=8000, total_demand=51028, daily_avg=851),
        "SOUTH": Node("SOUTH", init=4800, cap=5000, total_demand=32540, daily_avg=542),
        "NORTH": Node("NORTH", init=6000, cap=6000, total_demand=45270, daily_avg=755),
    }

    factory = Factory(
        init=5000,
        cap=6000,
        prod_daily=2970,
        prod_end=date(2025,2,26),
        allow_overcap=True,
        overcap_fee=5.0
    )

    start_date = date(2025,1,1)

    # ===== 日本：空运/海运 =====
    rt_jp_air = RouteTemplate(
        name="JP_AIR",
        dest="JP",
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
        ]
    )

    rt_jp_sea = RouteTemplate(
        name="JP_SEA",
        dest="JP",
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
        ]
    )

    # ===== 南方：直达/铁路 =====
    rt_south_direct = RouteTemplate(
        name="SOUTH_TRUCK_DIRECT",
        dest="SOUTH",
        legs=[
            ("宁波工厂->南方销售中心", [
                Carrier("开源陆运", 1, 16.42, 2000, 122, "CNY"),
                Carrier("易达快运", 1, 14.36, 6000, 418, "CNY"),
                Carrier("城市配送", 2, 23.26,  500,  22, "CNY"),
            ])
        ]
    )

    rt_south_rail = RouteTemplate(
        name="SOUTH_VIA_RAIL",
        dest="SOUTH",
        legs=[
            ("宁波工厂->宁波火车站", [
                Carrier("开源陆运", 1, 0.82, 2000, 2440, "CNY"),
                Carrier("易达快运", 1, 0.71, 6000, 8451, "CNY"),
                Carrier("城市配送", 1, 1.16,  500,  432, "CNY"),
            ]),
            ("宁波火车站->南京火车站(铁路)", [
                Carrier("西铁货运", 2, 8.10, 10000, 1235, "CNY"),
            ]),
            ("南京火车站->南方销售中心", [
                Carrier("开源陆运", 1, 1.20, 2000, 1667, "CNY"),
                Carrier("易达快运", 1, 1.05, 6000, 5715, "CNY"),
                Carrier("城市配送", 1, 1.70,  500,  295, "CNY"),
            ]),
        ]
    )

    # ===== 北方：直达/铁路 =====
    rt_north_direct = RouteTemplate(
        name="NORTH_TRUCK_DIRECT",
        dest="NORTH",
        legs=[
            ("宁波工厂->北方销售中心", [
                Carrier("开源陆运", 2, 57.22, 2000, 35, "CNY"),
                Carrier("易达快运", 3, 50.06, 6000, 120, "CNY"),
                Carrier("城市配送", 4, 81.06,  500,  7, "CNY"),
            ])
        ]
    )

    rt_north_rail = RouteTemplate(
        name="NORTH_VIA_RAIL",
        dest="NORTH",
        legs=[
            ("宁波工厂->宁波火车站", [
                Carrier("开源陆运", 1, 0.82, 2000, 2440, "CNY"),
                Carrier("易达快运", 1, 0.71, 6000, 8451, "CNY"),
                Carrier("城市配送", 1, 1.16,  500,  432, "CNY"),
            ]),
            ("宁波火车站->北京火车站(铁路)", [
                Carrier("西铁货运", 5, 26.71, 10000, 375, "CNY"),
            ]),
            ("北京火车站->北方销售中心", [
                Carrier("开源陆运", 1, 1.39, 2000, 1439, "CNY"),
                Carrier("易达快运", 1, 1.22, 6000, 4919, "CNY"),
                Carrier("城市配送", 1, 1.97,  500,  254, "CNY"),
            ]),
        ]
    )

    route_templates = [
        rt_jp_air, rt_jp_sea,
        rt_south_direct, rt_south_rail,
        rt_north_direct, rt_north_rail
    ]

    shipments, fac_df, node_dfs, summary = plan_all_nodes_with_factory_overcap_tradeoff_v12(
        start_date=start_date,
        nodes=nodes,
        factory=factory,
        route_templates=route_templates,
        fx=fx,
        max_repair_iters=2500,
        max_improve_iters=700,
        search_back_days=25
    )

    print("\n===== 三地联动 v1.2（含工厂超限费权衡）：摘要 =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    df_route = route_plan_df(shipments)
    df_leg = leg_plan_df(shipments, fx)

    fac_df.to_csv(here("factory_daily.csv"), index=False, encoding="utf-8-sig")
    node_dfs["JP"].to_csv(here("jp_daily.csv"), index=False, encoding="utf-8-sig")
    node_dfs["SOUTH"].to_csv(here("south_daily.csv"), index=False, encoding="utf-8-sig")
    node_dfs["NORTH"].to_csv(here("north_daily.csv"), index=False, encoding="utf-8-sig")
    df_route.to_csv(here("all_route_plan.csv"), index=False, encoding="utf-8-sig")
    df_leg.to_csv(here("all_leg_plan.csv"), index=False, encoding="utf-8-sig")

    print(f"\n[输出] factory_daily.csv -> {here('factory_daily.csv')}")
    print(f"[输出] jp_daily.csv      -> {here('jp_daily.csv')}")
    print(f"[输出] south_daily.csv   -> {here('south_daily.csv')}")
    print(f"[输出] north_daily.csv   -> {here('north_daily.csv')}")
    print(f"[输出] all_route_plan.csv -> {here('all_route_plan.csv')}")
    print(f"[输出] all_leg_plan.csv   -> {here('all_leg_plan.csv')}")

    # 画图
    plot_node(fac_df, cap=factory.cap, title="宁波工厂库存曲线（三地联动 v1.2，含超限费）", filename="factory_inventory.png", bar_col="发货")
    plot_node(node_dfs["JP"], cap=nodes["JP"].cap, title="日本销售中心库存曲线（三地联动 v1.2）", filename="jp_inventory.png", bar_col="到货")
    plot_node(node_dfs["SOUTH"], cap=nodes["SOUTH"].cap, title="南方销售中心库存曲线（三地联动 v1.2）", filename="south_inventory.png", bar_col="到货")
    plot_node(node_dfs["NORTH"], cap=nodes["NORTH"].cap, title="北方销售中心库存曲线（三地联动 v1.2）", filename="north_inventory.png", bar_col="到货")

    print("\n==== 完成：所有文件已输出到脚本所在目录 ====")
