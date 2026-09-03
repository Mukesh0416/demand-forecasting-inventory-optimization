"""AI-Powered Demand Forecasting & Inventory Optimization - dashboard.

Milestone 13 presentation layer. All figures come from the processed
artifacts produced by the Milestone-11 evaluation audit and the
Milestone-12 inventory optimization engine. Nothing is recalculated here.

Run with:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils import (
    get_recommended_policy,
    get_scenario_row,
    load_abc_performance,
    load_abc_series,
    load_audited_comparison,
    load_business_cases,
    load_eval_population,
    load_policy,
    load_recommendations,
    load_scenarios,
)

st.set_page_config(
    page_title="Demand Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide",
)

DEV_SUBSET_NOTE = (
    "Development subset: 300 HOBBIES_1 item-store series across CA_1, CA_2 and CA_3."
)
ASSUMPTION_NOTE = (
    "Inventory lead times, service levels, starting inventory and replenishment "
    "quantities are **scenario assumptions** because M5 does not provide these "
    "operational variables."
)

REC = get_recommended_policy()
SLS = [0.90, 0.95, 0.99]
LTS = [3, 7, 14]


def header(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def recommendation_banner() -> None:
    val = get_scenario_row("validation", REC["service_level"], REC["lead_time"])
    st.success(
        f"**RECOMMENDED POLICY** — {REC['service_level'] * 100:.0f}% Service Level "
        f"· {REC['lead_time']}-Day Lead Time  \n"
        f"Selected on **validation** performance only (weighted fill rate "
        f"{val['service_level_actual_weighted']:.3f} at "
        f"{val['average_inventory']:,.0f} average inventory units). "
        f"Test results were never used to select this scenario."
    )


def assumption_callout() -> None:
    st.info(ASSUMPTION_NOTE)


# ---------------------------------------------------------------------------
# Page 1 - Executive Overview
# ---------------------------------------------------------------------------
def page_executive_overview() -> None:
    header("Executive Overview",
           "Supply-chain analytics summary for the development subset")

    val = get_scenario_row("validation", REC["service_level"], REC["lead_time"])
    recommendation_banner()
    st.write(f"**{DEV_SUBSET_NOTE}**")

    kpi_row([
        ("Total Demand (validation)", f"{val['total_demand']:,.0f} units",
         "Actual demand in the validation window"),
        ("Number of Series", "300", "HOBBIES_1 item-store combinations"),
        ("Recommended Service Level", f"{REC['service_level'] * 100:.0f}%",
         "Scenario target of the recommended policy"),
        ("Recommended Lead Time", f"{REC['lead_time']} days",
         "Scenario assumption of the recommended policy"),
    ])
    kpi_row([
        ("Average Inventory", f"{val['average_inventory']:,.0f} units",
         "Total average on-hand inventory across all series = sum of per-series daily means (simulated)"),
        ("Weighted Fill Rate", f"{val['service_level_actual_weighted']:.1%}",
         "Share of demand served in the simulation (validation)"),
        ("Stockout Rate", f"{val['stockout_rate']:.2%}",
         "Days with unmet demand / all days (validation)"),
        ("Inventory Turnover", f"{val['inventory_turnover']:.2f}",
         "Simulated portfolio turnover (total demand / average inventory); scenario metric, not a real figure"),
    ])

    st.divider()
    st.subheader("Executive summary")
    st.markdown(
        "- **MA-28 is the selected primary forecasting method.** It won the "
        "audited validation comparison (MAE 1.0820); the Random Forest and "
        "XGBoost challengers did not provide a robust validation improvement.\n"
        "- The **inventory policy** converts the MA-28 forecast into safety "
        "stock and reorder points, backtested with a lead-time-respecting "
        "simulation.\n"
        f"- Within the modeled scenario space, **{REC['service_level'] * 100:.0f}% "
        f"service level / {REC['lead_time']}-day lead time** is the recommended "
        "scenario: it met the >= 98% weighted fill target with the lowest "
        "inventory among qualifying Pareto-frontier scenarios.\n"
        "- **ABC analysis shows strong demand concentration**: A-class series "
        "(96 of 300) contribute ~79.7% of total demand."
    )

    st.subheader("Audited model comparison (Milestone 11)")
    st.caption(
        "MA-28 is the selected primary forecasting model. Random Forest and "
        "XGBoost are challenger models, not the production forecast."
    )
    aud = load_audited_comparison()
    left, right = st.columns(2)
    with left:
        st.markdown("**Validation (selection basis)**")
        st.dataframe(
            aud[aud["split"] == "validation"].set_index("model")
            [["MAE", "RMSE", "WAPE"]].round(4), use_container_width=True)
    with right:
        st.markdown("**Test (final evaluation only)**")
        st.dataframe(
            aud[aud["split"] == "test"].set_index("model")
            [["MAE", "RMSE", "WAPE"]].round(4), use_container_width=True)

    assumption_callout()


# ---------------------------------------------------------------------------
# Page 2 - Demand Forecasting
# ---------------------------------------------------------------------------
def page_demand_forecasting() -> None:
    header("Demand Forecasting",
           "MA-28 forecast vs actual demand — audited Milestone-11 results")

    ep = load_eval_population()
    stores = sorted(ep["store_id"].unique())
    items = sorted(ep["item_id"].unique())

    c1, c2, c3 = st.columns(3)
    store = c1.selectbox("Store", stores, index=0)
    item = c2.selectbox(
        "Item", items,
        index=items.index("HOBBIES_1_074") if "HOBBIES_1_074" in items else 0)
    split = c3.radio("Period", ["validation", "test"], horizontal=True)

    sub = ep[(ep["store_id"] == store) & (ep["item_id"] == item)
             & (ep["split"] == split)].sort_values("date")
    sid = f"{item}_{store}"
    if sub.empty:
        st.warning("No data for this selection.")
        return

    st.markdown(
        f"#### {sid} — {split} window "
        f"({sub['date'].min():%Y-%m-%d} to {sub['date'].max():%Y-%m-%d})")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["demand"],
                             name="Actual demand", mode="lines",
                             line=dict(color="#2b6a99", width=1.4)))
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["ma_28_prediction"],
                             name="MA-28 forecast (primary)", mode="lines",
                             line=dict(color="#e45756", width=1.6)))
    if st.checkbox("Overlay challenger forecasts (Random Forest / XGBoost)"):
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["random_forest_prediction"],
                                 name="Random Forest (challenger)", mode="lines",
                                 line=dict(color="#8c8c8c", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=sub["date"], y=sub["xgboost_prediction"],
                                 name="XGBoost (challenger)", mode="lines",
                                 line=dict(color="#bcbd22", width=1, dash="dot")))
    fig.update_layout(height=400, yaxis_title="units/day",
                      title=f"Actual demand vs MA-28 forecast — {sid}",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    st.success("**MA-28 is the selected primary forecasting model.** It won the "
               "audited validation comparison; the ML models are challengers "
               "and are not the production forecast.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Audited metrics (all models, both windows)**")
        st.dataframe(
            load_audited_comparison()[["model", "split", "MAE", "RMSE", "WAPE"]]
            .sort_values(["split", "MAE"]).round(4),
            use_container_width=True, hide_index=True)
    with right:
        st.markdown(f"**Demand behavior indicators — {sid}** (selected window)")
        kpi_row([
            ("Zero-demand rate", f"{(sub['demand'] == 0).mean():.1%}", None),
            ("Mean demand", f"{sub['demand'].mean():.2f}", None),
            ("Std of demand", f"{sub['demand'].std():.2f}", None),
        ])
        st.caption(
            "A high zero-demand rate (intermittent demand) is the main reason "
            "MA-28 remains hard to beat on this subset.")

# ---------------------------------------------------------------------------
# Page 3 - Inventory Optimization
# ---------------------------------------------------------------------------
def page_inventory_optimization() -> None:
    header("Inventory Optimization",
           "Safety stock and reorder points from the MA-28 forecast "
           "(Milestone-12 engine, not recalculated here)")

    c1, c2 = st.columns(2)
    sl = c1.select_slider("Service Level (scenario)",
                          options=SLS, format_func=lambda v: f"{v * 100:.0f}%")
    lt = c2.select_slider("Lead Time in days (scenario)", options=LTS)

    val = get_scenario_row("validation", sl, lt)
    test = get_scenario_row("test", sl, lt)
    assumption_callout()

    st.markdown("#### Scenario KPIs (validation window)")
    kpi_row([
        ("Weighted Fill Rate", f"{val['service_level_actual_weighted']:.1%}",
         "Share of demand served in the simulation"),
        ("Average Inventory", f"{val['average_inventory']:,.0f} units",
         "Total average on-hand inventory across series = sum of per-series daily means"),
        ("Maximum Inventory", f"{val['maximum_inventory']:,.0f} units",
         "Max on-hand inventory across series"),
        ("Inventory Turnover", f"{val['inventory_turnover']:.2f}",
         "Simulated portfolio turnover: total demand / average inventory"),
    ])
    kpi_row([
        ("Stockout Units", f"{val['stockout_units']:,.0f}",
         "Demand not fulfilled (lost sales)"),
        ("Stockout Days", f"{val['stockout_days']:,.0f}",
         "Days with demand > 0 and unmet units"),
        ("Stockout Rate", f"{val['stockout_rate']:.2%}",
         "Stockout days / all days"),
        ("Mean Safety Stock", f"{val['safety_stock_mean']:.2f} units/series",
         "Z x sigma x sqrt(LT), sigma from train errors only"),
    ])

    if (sl, lt) == (REC["service_level"], REC["lead_time"]):
        recommendation_banner()
    st.caption(f"Test window (final evaluation only): weighted fill rate "
               f"{test['service_level_actual_weighted']:.1%}, stockout rate "
               f"{test['stockout_rate']:.2%}. Test is never used for selection.")

    st.markdown("#### Policy formulas")
    st.markdown(
        "**Safety stock protects against forecast uncertainty during the "
        "replenishment lead time.**\n\n"
        "- `Safety Stock = Z x sigma(forecast error) x sqrt(lead time)`\n"
        "- `ROP = expected lead-time demand + safety stock = "
        "forecast x lead time + safety stock`\n\n"
        "Sigma is the standard deviation of the MA-28 forecast error estimated "
        "on the **training period only** (<= 2015-04-13).")

    st.markdown("#### Reorder point detail — example series (validation)")
    ep = load_eval_population()
    policy = load_policy()
    example_sid = "HOBBIES_1_074_CA_3"
    pol = policy[(policy["id"] == example_sid) & (policy["service_level"] == sl)
                 & (policy["lead_time"] == lt)].sort_values("date")
    if not pol.empty:
        dem = ep[(ep["id"] == example_sid) & (ep["split"] == "validation")]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pol["date"], y=pol["reorder_point"],
                                 name="Reorder point", line=dict(width=1.8)))
        fig.add_trace(go.Scatter(x=pol["date"], y=pol["lead_time_demand"],
                                 name="Lead-time demand",
                                 line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=pol["date"], y=pol["safety_stock"],
                                 name="Safety stock",
                                 line=dict(width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=dem["date"], y=dem["demand"], name="Actual demand",
                                 line=dict(color="#bbbbbb", width=1), opacity=0.6))
        fig.update_layout(height=360, title=f"{example_sid} — policy components "
                          f"({sl * 100:.0f}% SL, {lt}-day LT)",
                          yaxis_title="units", legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Methodology notes"):
        st.markdown(
            "- Continuous-review (s, Q) policy: order when inventory position "
            "(on hand + on order) <= ROP.\n"
            "- Order quantity Q = max(1, ceil(forecast x lead time)) — scenario "
            "assumption.\n"
            "- Starting inventory = ROP on day 0 — simulation assumption.\n"
            "- Unmet demand is lost sales, not backordered — scenario assumption.\n"
            "- Full methodology: `docs/INVENTORY_METHODOLOGY.md`.")

# ---------------------------------------------------------------------------
# Page 4 - Scenario Analysis
# ---------------------------------------------------------------------------
def page_scenario_analysis() -> None:
    header("Scenario Analysis",
           "3 x 3 service-level x lead-time grid — validation is the selection "
           "basis, test is final evaluation only")

    sc = load_scenarios()
    val = sc[sc["window"] == "validation"].set_index(["service_level", "lead_time"])
    test = sc[sc["window"] == "test"].set_index(["service_level", "lead_time"])

    assumption_callout()

    for title, col, fmt in [
        ("Weighted fill rate", "service_level_actual_weighted", "{:.1%}"),
        ("Average inventory (units)", "average_inventory", "{:,.0f}"),
        ("Stockout rate", "stockout_rate", "{:.2%}"),
    ]:
        st.markdown(f"#### {title} — validation (selection basis)")
        grid = val[col].unstack("lead_time")[LTS].reindex(SLS)
        grid.index = [f"{int(i * 100)}% SL" for i in grid.index]
        grid.columns = [f"{int(c)}d" for c in grid.columns]
        styled = grid.map(lambda v: fmt.format(v))
        rec_label = (f"{int(REC['service_level'] * 100)}% SL", f"{REC['lead_time']}d")
        if rec_label[0] in styled.index and rec_label[1] in styled.columns:
            styled.loc[rec_label] = styled.loc[rec_label].map(lambda v: f"**{v} ★**")
        st.dataframe(styled, use_container_width=True)

    st.markdown("#### Trade-off charts (validation)")
    palette = {3: "#2b6a99", 7: "#e4a11b", 14: "#e45756"}
    fig = go.Figure()
    for lt in LTS:
        sub = val.xs(lt, level="lead_time").reindex(SLS)
        fig.add_trace(go.Scatter(
            x=[s * 100 for s in SLS], y=sub["average_inventory"],
            name=f"LT={lt}d", mode="lines+markers", line=dict(color=palette[lt]),
            customdata=sub["service_level_actual_weighted"],
            hovertemplate=("SL %{x:.0f}% — avg inv %{y:,.0f} — fill "
                           f"%{{customdata:.1%}}<extra>LT={lt}d</extra>")))
    rec_xy = (REC["service_level"] * 100,
              val.loc[(REC["service_level"], REC["lead_time"]), "average_inventory"])
    fig.add_trace(go.Scatter(x=[rec_xy[0]], y=[rec_xy[1]], mode="markers+text",
                             text=["RECOMMENDED"], textposition="top center",
                             marker=dict(size=14, color="#2e7d32", symbol="star"),
                             name="Recommended"))
    fig.update_layout(height=380, title="Service level vs average inventory",
                      xaxis_title="target service level (%)",
                      yaxis_title="average inventory (units)")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = go.Figure()
        for sl in SLS:
            sub = val.xs(sl, level="service_level").reindex(LTS)
            fig2.add_trace(go.Scatter(x=LTS, y=sub["average_inventory"],
                                      name=f"SL={sl * 100:.0f}%", mode="lines+markers"))
        fig2.update_layout(height=340, title="Lead time vs average inventory",
                           xaxis_title="lead time (days)",
                           yaxis_title="average inventory (units)")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = go.Figure()
        for sl in SLS:
            sub = val.xs(sl, level="service_level").reindex(LTS)
            fig3.add_trace(go.Scatter(x=LTS, y=sub["service_level_actual_weighted"],
                                      name=f"SL={sl * 100:.0f}%", mode="lines+markers"))
        fig3.update_layout(height=340,
                           title="Lead time vs weighted fill rate (validation)",
                           xaxis_title="lead time (days)", yaxis_title="weighted fill rate")
        st.plotly_chart(fig3, use_container_width=True)

    st.success(f"**Recommended scenario (validation-selected): {REC['service_level'] * 100:.0f}% SL "
               f"/ {REC['lead_time']}-day LT** — met the >= 98% weighted fill target "
               "with the lowest inventory among qualifying scenarios.")
    with st.expander("Test window — final evaluation only (never used for selection)"):
        tgrid = test["service_level_actual_weighted"].unstack("lead_time")[LTS].reindex(SLS)
        tgrid.index = [f"{int(i * 100)}% SL" for i in tgrid.index]
        tgrid.columns = [f"{int(c)}d" for c in tgrid.columns]
        st.dataframe(tgrid.map("{:.1%}".format), use_container_width=True)

# ---------------------------------------------------------------------------
# Page 5 - ABC Analysis
# ---------------------------------------------------------------------------
def page_abc_analysis() -> None:
    header("ABC Analysis",
           "Analytical demand-contribution classification of the development "
           "subset — not an official Walmart ABC policy")

    abc = load_abc_series()
    perf = load_abc_performance()
    assumption_callout()

    counts = abc.groupby("abc_class")["base_id"].nunique().reindex(["A", "B", "C"])
    contrib = abc.groupby("abc_class")["demand_percentage"].sum().reindex(["A", "B", "C"])

    kpi_row([
        (f"{cls}-class series", f"{counts[cls]:.0f}",
         f"{contrib[cls]:.1%} of total demand")
        for cls in ["A", "B", "C"]
    ])

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Pareto — cumulative demand contribution")
        one = abc.drop_duplicates("base_id").sort_values(
            "cumulative_demand_percentage", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(1, len(one) + 1)),
            y=one["demand_percentage"] * 100, name="Demand share (%)",
            marker_color="#2b6a99", opacity=0.7))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(one) + 1)),
            y=one["cumulative_demand_percentage"] * 100,
            name="Cumulative demand (%)", line=dict(color="#e45756", width=2)))
        for cut, lbl in [(80, "80% (A)"), (95, "95% (A+B)")]:
            fig.add_hline(y=cut, line_dash="dot", line_color="#888",
                          annotation_text=lbl)
        fig.update_layout(height=380, xaxis_title="series rank (high -> low demand)",
                          yaxis_title="% of total demand",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Series counts by class")
        figc = go.Figure(go.Bar(
            x=counts.index.astype(str), y=counts.values,
            marker_color=["#2e7d32", "#e4a11b", "#9e9e9e"]))
        figc.update_layout(height=380, xaxis_title="ABC class",
                           yaxis_title="number of series")
        st.plotly_chart(figc, use_container_width=True)

    abc_filter = st.multiselect("Filter by ABC class", ["A", "B", "C"],
                                default=["A", "B", "C"])
    st.markdown("#### Series detail (filtered)")
    view = abc[abc["abc_class"].isin(abc_filter)][
        ["base_id", "abc_class", "total_demand", "demand_percentage",
         "cumulative_demand_percentage"]].rename(
        columns={"base_id": "id"}).sort_values(
        ["abc_class", "total_demand"], ascending=[True, False])
    st.dataframe(
        view.assign(demand_percentage=view["demand_percentage"].map("{:.3%}".format),
                    cumulative_demand_percentage=view[
                        "cumulative_demand_percentage"].map("{:.2%}".format)),
        use_container_width=True, hide_index=True, height=320)

    st.markdown("#### Inventory performance by class (validation, 95% SL / 7d)")
    vp = perf[(perf["window"] == "validation")
              & (perf["service_level"] == 0.95) & (perf["lead_time"] == 7)]
    st.dataframe(
        vp[["abc_class", "total_demand", "stockout_rate", "average_inventory",
            "inventory_turnover"]].sort_values("abc_class").assign(
            total_demand=lambda d: d["total_demand"].map("{:,.0f}".format),
            stockout_rate=lambda d: d["stockout_rate"].map("{:.2%}".format),
            average_inventory=lambda d: d["average_inventory"].map("{:,.0f}".format),
            inventory_turnover=lambda d: d["inventory_turnover"].map("{:.2f}".format)),
        use_container_width=True, hide_index=True)

    st.markdown(
        "**Interpretation:** A-class series represent the majority of demand and "
        "deserve tighter monitoring, higher service-level targets and frequent "
        "replenishment review. C-class series are lower contribution and often "
        "more intermittent, so excessive buffers should be avoided. This is an "
        "**analytical classification of this project's development subset**, "
        "not an official Walmart ABC policy.")

# ---------------------------------------------------------------------------
# Page 6 - Business Cases
# ---------------------------------------------------------------------------
def page_business_cases() -> None:
    header("Business Cases",
           "Three representative demand profiles and how the inventory policy "
           "differs for each")

    bc = load_business_cases()
    bc = bc[bc["window"] == "validation"].sort_values("case").set_index("case")
    label = {
        "stable_high_demand": "1. Stable / High Demand",
        "intermittent": "2. Intermittent Demand",
        "volatile": "3. Volatile Demand",
    }
    case = st.selectbox("Select demand profile", list(bc.index),
                        format_func=lambda c: label.get(c, c))
    row = bc.loc[case]
    assumption_callout()

    st.markdown(f"#### {label.get(case, case)} — `{row['id'].replace('_validation', '')}`")
    kpi_row([
        ("Mean demand", f"{row['mean_demand']:.2f} units/day", None),
        ("Demand std", f"{row['std_demand']:.2f}", "Day-to-day variability"),
        ("Mean forecast", f"{row['mean_forecast']:.2f} units/day",
         "MA-28 forecast (validation)"),
        ("Actual service level", f"{row['service_level_actual']:.1%}",
         "Realized fill rate in the simulation"),
    ])
    kpi_row([
        ("Lead time", f"{int(row['lead_time'])} days", "Scenario assumption"),
        ("Service level target", f"{row['service_level'] * 100:.0f}%",
         "Scenario assumption"),
        ("Safety stock", f"{row['mean_safety_stock']:.1f} units",
         "Z x sigma x sqrt(LT)"),
        ("Reorder point", f"{row['mean_reorder_point']:.1f} units",
         "Lead-time demand + safety stock"),
    ])
    kpi_row([
        ("Average inventory", f"{row['average_inventory']:.1f} units",
         "Simulated mean on-hand"),
        ("Stockout rate", f"{row['stockout_rate']:.2%}",
         "Days with unmet demand / all days"),
    ])

    st.markdown("#### Comparison across profiles")
    comp = bc.rename(index=lambda c: label.get(c, c))
    fig = go.Figure()
    for metric, nice in [("mean_safety_stock", "Safety stock"),
                         ("mean_reorder_point", "Reorder point"),
                         ("average_inventory", "Average inventory")]:
        fig.add_trace(go.Bar(name=nice, x=list(comp.index.astype(str)),
                             y=comp[metric]))
    fig.add_trace(go.Scatter(name="Stockout rate (right axis)",
                             x=list(comp.index.astype(str)),
                             y=comp["stockout_rate"], yaxis="y2",
                             mode="lines+markers", line=dict(color="#e45756")))
    fig.update_layout(height=380, barmode="group",
                      yaxis=dict(title="units"),
                      yaxis2=dict(title="stockout rate", overlaying="y",
                                  side="right", tickformat=".1%", showgrid=False),
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Why the policies differ:** stable, high-demand series are predictable, "
        "so a modest safety stock suffices but stock must always be present — "
        "their inventory cost is dominated by cycle stock. Intermittent series "
        "have many zero-demand days: conventional (s, Q) policies tend to hold "
        "dead buffer stock, so lower targets and closer review are advisable. "
        "Volatile series have high demand std, which the safety-stock formula "
        "translates into the largest buffers — these benefit most from better "
        "forecasting and from shorter lead times.")

# ---------------------------------------------------------------------------
# Page 7 - Recommendations
# ---------------------------------------------------------------------------
def page_recommendations() -> None:
    header("Recommendations",
           "Analytical conclusions from the development subset and scenario "
           "simulation — not actual Walmart operational policy")

    recs = load_recommendations()
    st.caption(
        "Every recommendation below is derived from the audited Milestone-11 "
        "model comparison and the Milestone-12 inventory simulation on the "
        "300-series development subset. They are analytical conclusions, not "
        "company policy."
    )

    recommendations_md = [
        ("1 — Use MA-28 as the primary forecast",
         "MA-28 won the audited validation comparison (MAE 1.0820 vs Random "
         "Forest 1.0860 and XGBoost 1.0897). The ML challengers beat MA-28 on "
         "fewer than half of individual series and their nominal test edge "
         "(+0.17%) is within noise. Keep Random Forest / XGBoost as challenger "
         "models for monitoring, not as the production forecast."),
        ("2 — Default scenario: 99% service level / 3-day lead time",
         "Within the modeled scenario space, this scenario met the >= 98% "
         "weighted fill-rate target (0.9884) with the lowest average inventory "
         "(2,827 units) among qualifying Pareto-frontier scenarios. Selected "
         "on validation performance only."),
        ("3 — Prioritize lead-time reduction",
         "At the 99% service level, moving from a 14-day to a 3-day lead time "
         "cuts average inventory by roughly 60% (7,120 -> 2,827 units) while "
         "improving the realized fill rate. Lead-time reduction dominates "
         "z-score increases as an inventory lever."),
        ("4 — Differentiated ABC inventory attention",
         "A-class series (96 of 300) contribute ~79.7% of demand and turn over "
         "~2x faster than C-class; they deserve tighter monitoring and "
         "frequent replenishment review. C-class series (114 of 300, ~5.1% of "
         "demand) should avoid heavy buffers."),
        ("5 — Treat intermittent-demand series separately",
         "Many series have a high zero-demand rate, which makes conventional "
         "(s, Q) policies hold dead buffer stock. Lower service-level targets "
         "and closer review are advisable for these series."),
    ]
    for title, body in recommendations_md:
        with st.expander(title, expanded=len(title) < 60):
            st.markdown(body)

    st.divider()
    st.subheader("Recommendation register (from processed results)")
    st.dataframe(recs, use_container_width=True)
    assumption_callout()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
PAGES = {
    "1. Executive Overview": page_executive_overview,
    "2. Demand Forecasting": page_demand_forecasting,
    "3. Inventory Optimization": page_inventory_optimization,
    "4. Scenario Analysis": page_scenario_analysis,
    "5. ABC Analysis": page_abc_analysis,
    "6. Business Cases": page_business_cases,
    "7. Recommendations": page_recommendations,
}


def main() -> None:
    with st.sidebar:
        st.markdown("## 📦 Demand & Inventory")
        st.caption(
            "M5 Retail Demand Planning | Forecasting | Inventory Policy | "
            "ABC Analysis"
        )
        st.divider()
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
        st.divider()
        st.caption(DEV_SUBSET_NOTE)
        st.caption(
            "Lead times, service levels, starting inventory and replenishment "
            "quantities are **scenario assumptions** (M5 provides no "
            "operational variables)."
        )

    header("AI-Powered Demand Forecasting & Inventory Optimization",
           "M5 Retail Demand Planning | Forecasting | Inventory Policy | "
           "ABC Analysis")
    st.divider()
    PAGES[page]()


if __name__ == "__main__":
    main()







