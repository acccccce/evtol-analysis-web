from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import root


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR
MODEL_DIR = PROJECT_DIR / "flight_dynamics_report"
DEFAULT_DOCX = PROJECT_DIR / "data" / "eVTOL三视图与参数.docx"
REPORT_DOCX = PROJECT_DIR / "reports" / "eVTOL巡航平飞性能与模态特性分析报告.docx"
TRANSITION_REPORT_DOCX = PROJECT_DIR / "reports" / "eVTOL悬停构型平飞性能与过渡飞行走廊分析报告_公式格式修订版.docx"

sys.path.insert(0, str(MODEL_DIR))
from evtol_flight_dynamics_analysis import (  # noqa: E402
    AircraftModel,
    G,
    MAX_ABS_TAIL_DEFLECTION_DEG,
    VALID_ALPHA_RANGE_DEG,
    calculate_lateral_modes,
    calculate_longitudinal_modes,
    isa_density,
    json_safe,
    solve_level_trim,
)


HEIGHT_COLORS = ["#17365D", "#2E75B6", "#2A9D8F", "#D9822B", "#8B5CF6", "#C94C4C"]
DEFAULT_TILT_ANGLES = [90, 75, 60, 45, 30, 15, 0]
DEFAULT_PROP_COUNT = 6
DEFAULT_PROP_DIAMETER_M = 3.0
DEFAULT_FORWARD_EFFICIENCY = 0.68

st.set_page_config(
    page_title="eVTOL 飞行动力学分析",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 3rem;}
      [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f7fafc 0%, #eef4f8 100%);
        border: 1px solid #d9e3ec; border-radius: 12px; padding: 12px 14px;
      }
      .hero {
        padding: 1.15rem 1.35rem; border-radius: 16px;
        background: linear-gradient(120deg, #17365D 0%, #2E75B6 100%);
        color: white; margin-bottom: 1rem;
      }
      .hero h1 {font-size: 1.85rem; margin: 0 0 .25rem 0;}
      .hero p {margin: 0; opacity: .88;}
      .status-ok, .status-bad {
        border-radius: 999px; display: inline-block; padding: .22rem .65rem;
        font-weight: 600;
      }
      .status-ok {color: #166534; background: #dcfce7;}
      .status-bad {color: #991b1b; background: #fee2e2;}
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_heights(text: str) -> list[int]:
    values = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(float(part))
        if value < 0 or value > 10000:
            raise ValueError("高度应位于 0～10000 m。")
        values.append(value)
    values = sorted(set(values))
    if not values:
        raise ValueError("至少需要一个高度。")
    return values


def parse_tilt_angles(text: str) -> list[int]:
    values = []
    for part in text.replace("，", ",").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(float(part))
        if value < 0 or value > 90:
            raise ValueError("推力倾角应位于 0～90 deg。")
        values.append(value)
    values = sorted(set(values), reverse=True)
    if not values:
        raise ValueError("至少需要一个推力倾角。")
    return values


def estimate_transition_power(
    rho: float,
    speed_mps: float,
    thrust_n: float,
    alpha_rad: float,
    tilt_rad: float,
    propeller_count: int,
    propeller_diameter_m: float,
    forward_efficiency: float,
) -> float:
    disk_area = propeller_count * math.pi * (propeller_diameter_m / 2.0) ** 2
    forward_power = max(0.0, thrust_n * speed_mps * math.cos(alpha_rad + tilt_rad))
    vertical_thrust = max(0.0, thrust_n * math.sin(tilt_rad))
    induced_power = (
        vertical_thrust ** 1.5 / math.sqrt(2.0 * rho * disk_area)
        if vertical_thrust > 0.0
        else 0.0
    )
    return (forward_power / forward_efficiency + induced_power) / 1000.0


def solve_tilted_level_trim(
    aircraft: AircraftModel,
    height_m: float,
    speed_mps: float,
    tilt_deg: float,
    propeller_count: int,
    propeller_diameter_m: float,
    forward_efficiency: float,
) -> dict:
    rho = isa_density(height_m)
    dynamic_pressure = 0.5 * rho * speed_mps**2
    tilt_rad = math.radians(tilt_deg)

    def residual(unknowns: np.ndarray) -> np.ndarray:
        alpha, delta_e, thrust = unknowns
        cl, cd, cm = aircraft.longitudinal_coefficients(alpha, delta_e)
        force_x = (
            dynamic_pressure
            * aircraft.S
            * (-cd * math.cos(alpha) + cl * math.sin(alpha))
            + thrust * math.cos(tilt_rad)
        )
        force_z = (
            dynamic_pressure
            * aircraft.S
            * (-cd * math.sin(alpha) - cl * math.cos(alpha))
            - thrust * math.sin(tilt_rad)
        )
        return np.array(
            [
                force_x / aircraft.mass - G * math.sin(alpha),
                force_z / aircraft.mass + G * math.cos(alpha),
                cm,
            ]
        )

    denominator = max(0.2, math.sin(tilt_rad) if tilt_deg > 1.0 else 0.2)
    best_solution = None
    best_norm = math.inf
    for alpha_guess_deg in [-2.0, 0.0, 4.0, 8.0, 12.0]:
        guess = np.array(
            [math.radians(alpha_guess_deg), 0.0, aircraft.weight / denominator]
        )
        solution = root(residual, guess, method="hybr", tol=1e-11)
        norm = float(np.linalg.norm(residual(solution.x)))
        if best_solution is None or (solution.success and norm < best_norm):
            best_solution = solution
            best_norm = norm

    alpha, delta_e, thrust = best_solution.x
    cl, cd, cm = aircraft.longitudinal_coefficients(alpha, delta_e)
    lift = dynamic_pressure * aircraft.S * cl
    drag = dynamic_pressure * aircraft.S * cd
    valid = bool(
        best_solution.success
        and VALID_ALPHA_RANGE_DEG[0] <= math.degrees(alpha) <= VALID_ALPHA_RANGE_DEG[1]
        and abs(delta_e) <= MAX_ABS_TAIL_DEFLECTION_DEG
        and thrust >= 0.0
        and cd > 0.0
    )
    return {
        "height_m": height_m,
        "speed_mps": speed_mps,
        "tilt_deg": tilt_deg,
        "rho_kg_m3": rho,
        "valid": valid,
        "alpha_deg": math.degrees(alpha),
        "delta_e_deg": float(delta_e),
        "thrust_total_N": float(thrust),
        "thrust_per_prop_N": float(thrust / propeller_count),
        "estimated_power_kW": estimate_transition_power(
            rho,
            speed_mps,
            thrust,
            alpha,
            tilt_rad,
            propeller_count,
            propeller_diameter_m,
            forward_efficiency,
        ),
        "CL": cl,
        "CD": cd,
        "lift_N": lift,
        "drag_N": drag,
        "vertical_balance_ratio": (
            lift + thrust * math.sin(tilt_rad) * math.cos(alpha)
        )
        / aircraft.weight,
        "residual_norm": best_norm,
        "solver_success": bool(best_solution.success),
    }


def run_transition_analysis(
    aircraft: AircraftModel,
    heights: list[int],
    hover_speeds: list[int],
    corridor_speeds: list[int],
    tilt_angles: list[int],
    propeller_count: int,
    propeller_diameter_m: float,
    forward_efficiency: float,
) -> dict:
    hover_rows = []
    for height in heights:
        for speed in hover_speeds:
            hover_rows.append(
                solve_tilted_level_trim(
                    aircraft,
                    height,
                    speed,
                    90,
                    propeller_count,
                    propeller_diameter_m,
                    forward_efficiency,
                )
            )

    tilt_rows = []
    for tilt in tilt_angles:
        for speed in corridor_speeds:
            tilt_rows.append(
                solve_tilted_level_trim(
                    aircraft,
                    0,
                    speed,
                    tilt,
                    propeller_count,
                    propeller_diameter_m,
                    forward_efficiency,
                )
            )

    hover_summary = []
    for height in heights:
        valid_rows = [
            row for row in hover_rows if row["height_m"] == height and row["valid"]
        ]
        if valid_rows:
            hover_summary.append(
                {
                    "height_m": height,
                    "min_speed_mps": min(row["speed_mps"] for row in valid_rows),
                    "max_speed_mps": max(row["speed_mps"] for row in valid_rows),
                    "min_thrust_N": min(row["thrust_total_N"] for row in valid_rows),
                    "max_thrust_N": max(row["thrust_total_N"] for row in valid_rows),
                    "min_power_kW": min(row["estimated_power_kW"] for row in valid_rows),
                    "max_power_kW": max(row["estimated_power_kW"] for row in valid_rows),
                }
            )

    corridor = []
    for tilt in tilt_angles:
        valid_rows = [
            row for row in tilt_rows if row["tilt_deg"] == tilt and row["valid"]
        ]
        corridor.append(
            {
                "tilt_deg": tilt,
                "min_speed_mps": min([row["speed_mps"] for row in valid_rows], default=None),
                "max_speed_mps": max([row["speed_mps"] for row in valid_rows], default=None),
                "min_thrust_N": min([row["thrust_total_N"] for row in valid_rows], default=None),
                "max_thrust_N": max([row["thrust_total_N"] for row in valid_rows], default=None),
                "min_power_kW": min([row["estimated_power_kW"] for row in valid_rows], default=None),
                "max_power_kW": max([row["estimated_power_kW"] for row in valid_rows], default=None),
            }
        )

    usable_corridor = [row for row in corridor if row["min_speed_mps"] is not None]
    if usable_corridor:
        common_min = max(row["min_speed_mps"] for row in usable_corridor)
        common_max = min(row["max_speed_mps"] for row in usable_corridor)
    else:
        common_min = None
        common_max = None

    return {
        "hover_rows": hover_rows,
        "tilt_rows": tilt_rows,
        "hover_summary": hover_summary,
        "corridor": corridor,
        "common_speed_range": (common_min, common_max),
        "settings": {
            "tilt_angles_deg": tilt_angles,
            "propeller_count": propeller_count,
            "propeller_diameter_m": propeller_diameter_m,
            "forward_efficiency": forward_efficiency,
        },
    }

def uploaded_docx_path(uploaded_file) -> Path:
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:12]
    cache_dir = APP_DIR / ".runtime"
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / f"uploaded_{digest}.docx"
    if not path.exists():
        path.write_bytes(uploaded_file.getvalue())
    return path



def run_analysis(
    source_path: Path,
    heights: list[int],
    speed_min: int,
    speed_max: int,
    speed_step: int,
    modal_speed: float,
    mass_override: float | None,
    include_transition: bool,
    transition_speed_min: int,
    transition_speed_max: int,
    transition_speed_step: int,
    tilt_angles: list[int],
    propeller_count: int,
    propeller_diameter_m: float,
    forward_efficiency: float,
) -> dict:
    aircraft = AircraftModel(source_path)
    if mass_override is not None:
        aircraft.mass = mass_override
        aircraft.weight = aircraft.mass * 9.80665

    performance = []
    speeds = list(range(speed_min, speed_max + 1, speed_step))
    for height in heights:
        guess = None
        for speed in speeds:
            trim = solve_level_trim(aircraft, height, speed, guess)
            guess = trim.pop("solution_vector")
            performance.append(trim)

    modal_rows = []
    linear_models = {}
    cruise_trims = []
    for height in heights:
        trim = solve_level_trim(aircraft, height, modal_speed)
        trim.pop("solution_vector")
        cruise_trims.append(trim)
        longitudinal = calculate_longitudinal_modes(aircraft, trim)
        lateral = calculate_lateral_modes(aircraft, trim)

        for mode in longitudinal["modes"]:
            modal_rows.append(
                {"height_m": height, "speed_mps": modal_speed, "axis": "纵向", **mode}
            )
        for mode in lateral["modes"]:
            modal_rows.append(
                {"height_m": height, "speed_mps": modal_speed, "axis": "横航向", **mode}
            )

        linear_models[height] = {
            "longitudinal_A": longitudinal["A"],
            "longitudinal_eigenvalues": longitudinal["eigenvalues"],
            "lateral_A": lateral["A"],
            "lateral_eigenvalues": lateral["eigenvalues"],
            "lateral_derivatives": lateral["derivatives"],
        }

    transition = None
    if include_transition:
        transition_speeds = list(
            range(transition_speed_min, transition_speed_max + 1, transition_speed_step)
        )
        hover_speeds = [speed for speed in transition_speeds if speed <= 80]
        if not hover_speeds:
            hover_speeds = transition_speeds
        transition = run_transition_analysis(
            aircraft,
            heights,
            hover_speeds,
            transition_speeds,
            tilt_angles,
            propeller_count,
            propeller_diameter_m,
            forward_efficiency,
        )

    return {
        "aircraft": aircraft,
        "heights": heights,
        "speeds": speeds,
        "modal_speed": modal_speed,
        "performance": performance,
        "modal_rows": modal_rows,
        "cruise_trims": cruise_trims,
        "linear_models": linear_models,
        "transition": transition,
    }


def performance_frame(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results["performance"]).drop(
        columns=["solver_success"], errors="ignore"
    )


def modal_frame(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results["modal_rows"])


def transition_frame(results: dict, key: str) -> pd.DataFrame:
    transition = results.get("transition")
    if not transition:
        return pd.DataFrame()
    return pd.DataFrame(transition[key])


def make_download_zip(results: dict) -> bytes:
    performance = performance_frame(results)
    modal = modal_frame(results)
    payload = {
        "settings": {
            "heights_m": results["heights"],
            "speeds_mps": results["speeds"],
            "modal_speed_mps": results["modal_speed"],
        },
        "performance": results["performance"],
        "modal_results": results["modal_rows"],
        "linear_models": results["linear_models"],
        "transition": results.get("transition"),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "level_flight_performance.csv",
            performance.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "modal_results.csv", modal.to_csv(index=False).encode("utf-8-sig")
        )
        if results.get("transition"):
            transition = results["transition"]
            archive.writestr(
                "hover_configuration_level_flight.csv",
                pd.DataFrame(transition["hover_rows"]).to_csv(index=False).encode("utf-8-sig"),
            )
            archive.writestr(
                "tilt_level_flight_scan.csv",
                pd.DataFrame(transition["tilt_rows"]).to_csv(index=False).encode("utf-8-sig"),
            )
            archive.writestr(
                "transition_corridor_summary.csv",
                pd.DataFrame(transition["corridor"]).to_csv(index=False).encode("utf-8-sig"),
            )
        archive.writestr(
            "analysis_results.json",
            json.dumps(payload, ensure_ascii=False, indent=2, default=json_safe),
        )
    return buffer.getvalue()


def style_plot(figure, title: str, x_title: str, y_title: str):
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
        height=440,
        margin=dict(l=35, r=20, t=65, b=35),
        legend_title_text="高度",
        hovermode="x unified",
    )
    return figure


st.markdown(
    """
    <div class="hero">
      <h1>eVTOL 飞行动力学分析工具</h1>
      <p>非线性定常平飞配平 · 需用推力与功率 · 纵向/横航向模态 · 悬停构型与过渡走廊</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("计算设置")
    source_mode = st.radio("数据来源", ["使用项目默认数据", "上传 Word 数据文件"])
    uploaded = None
    if source_mode == "上传 Word 数据文件":
        uploaded = st.file_uploader("选择 .docx 文件", type=["docx"])

    height_text = st.text_input("分析高度（m，逗号分隔）", "0, 1000, 2000, 3000")
    speed_range = st.slider("性能速度范围（m/s）", 30, 140, (40, 110), step=5)
    speed_step = st.select_slider("速度步长（m/s）", options=[1, 2, 5, 10], value=5)
    modal_speed = st.number_input(
        "模态基准速度（m/s）", 30.0, 140.0, 65.0, step=1.0
    )
    override_mass = st.checkbox("修改计算质量")
    mass_override = (
        st.number_input("质量（kg）", 100.0, 10000.0, 2255.52, step=10.0)
        if override_mass
        else None
    )

    st.divider()
    st.subheader("过渡走廊设置")
    include_transition = st.checkbox("计算悬停构型与过渡走廊", value=True)
    transition_speed_range = st.slider(
        "过渡速度范围（m/s）", 5, 120, (5, 110), step=5
    )
    transition_speed_step = st.select_slider(
        "过渡速度步长（m/s）", options=[1, 2, 5, 10], value=5
    )
    tilt_text = st.text_input(
        "推力倾角 beta（deg，逗号分隔）", "90, 75, 60, 45, 30, 15, 0"
    )
    propeller_count = st.number_input(
        "升力桨数量", 1, 16, DEFAULT_PROP_COUNT, step=1
    )
    propeller_diameter = st.number_input(
        "桨径 D（m）", 0.5, 8.0, DEFAULT_PROP_DIAMETER_M, step=0.1
    )
    forward_efficiency = st.number_input(
        "前向推进效率 eta_p", 0.10, 0.95, DEFAULT_FORWARD_EFFICIENCY, step=0.01
    )

    calculate = st.button("开始计算", type="primary", use_container_width=True)
    st.caption("巡航功率为 D·V；过渡段功率为前向有效功率 + 垂向诱导功率估算。")

if calculate:
    try:
        heights = parse_heights(height_text)
        if source_mode == "上传 Word 数据文件":
            if uploaded is None:
                raise ValueError("请先上传 Word 数据文件。")
            source_path = uploaded_docx_path(uploaded)
        else:
            source_path = DEFAULT_DOCX
        tilt_angles = parse_tilt_angles(tilt_text)
        with st.spinner("正在求解非线性配平、模态和过渡走廊……"):
            st.session_state["analysis_results"] = run_analysis(
                source_path,
                heights,
                speed_range[0],
                speed_range[1],
                speed_step,
                modal_speed,
                mass_override,
                include_transition,
                transition_speed_range[0],
                transition_speed_range[1],
                transition_speed_step,
                tilt_angles,
                propeller_count,
                propeller_diameter,
                forward_efficiency,
            )
            st.session_state["source_name"] = source_path.name
    except Exception as exc:
        st.error(f"计算失败：{exc}")

if "analysis_results" not in st.session_state:
    st.info("使用左侧默认设置点击“开始计算”，即可复现报告中的分析结果。")
    st.markdown(
        """
        **工具将完成：**

        1. 求解迎角、对称尾翼舵偏和推力三个未知量；
        2. 绘制不同高度的需用推力、功率和配平变量曲线；
        3. 对巡航配平点进行纵向与横航向数值线化；
        4. 识别长周期、短周期、荷兰滚、滚转和螺旋模态；
        5. 计算悬停构型平飞性能和不同倾角过渡走廊；
        6. 导出 CSV 与 JSON 结果。
        """
    )
    st.stop()

results = st.session_state["analysis_results"]
performance = performance_frame(results)
modal = modal_frame(results)
valid_performance = performance[performance["valid"]].copy()
cruise = pd.DataFrame(results["cruise_trims"])

tabs = st.tabs(["总览", "平飞性能", "过渡走廊", "模态分析", "状态矩阵", "结果导出"])

with tabs[0]:
    st.subheader("巡航配平总览")
    st.caption(
        f"数据文件：{st.session_state.get('source_name', DEFAULT_DOCX.name)}；"
        f"模态基准速度：{results['modal_speed']:.1f} m/s"
    )
    selected = cruise.iloc[0]
    metric_cols = st.columns(5)
    metric_cols[0].metric("质量", f"{results['aircraft'].mass:.1f} kg")
    metric_cols[1].metric("迎角", f"{selected['alpha_deg']:.3f}°")
    metric_cols[2].metric("尾翼舵偏", f"{selected['delta_e_deg']:.3f}°")
    metric_cols[3].metric("需用推力", f"{selected['thrust_N']:.1f} N")
    metric_cols[4].metric("需用功率", f"{selected['power_required_kW']:.2f} kW")

    summary = cruise[
        [
            "height_m",
            "alpha_deg",
            "delta_e_deg",
            "thrust_N",
            "power_required_kW",
            "lift_weight_ratio",
            "thrust_drag_ratio",
            "residual_norm",
        ]
    ].rename(
        columns={
            "height_m": "高度 (m)",
            "alpha_deg": "迎角 (°)",
            "delta_e_deg": "尾翼舵偏 (°)",
            "thrust_N": "需用推力 (N)",
            "power_required_kW": "需用功率 (kW)",
            "lift_weight_ratio": "L/W",
            "thrust_drag_ratio": "T/D",
            "residual_norm": "方程残差",
        }
    )
    st.dataframe(
        summary.style.format(
            {
                "迎角 (°)": "{:.3f}",
                "尾翼舵偏 (°)": "{:.3f}",
                "需用推力 (N)": "{:.1f}",
                "需用功率 (kW)": "{:.2f}",
                "L/W": "{:.5f}",
                "T/D": "{:.5f}",
                "方程残差": "{:.2e}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    unstable = modal[~modal["stable"]]
    if unstable.empty:
        st.markdown('<span class="status-ok">所有模态稳定</span>', unsafe_allow_html=True)
    else:
        names = "、".join(sorted(unstable["mode"].unique()))
        st.markdown(
            f'<span class="status-bad">存在不稳定模态：{names}</span>',
            unsafe_allow_html=True,
        )
        st.warning("当前数据下螺旋模态呈弱发散；可进一步设计滚转—航向增稳控制。")

with tabs[1]:
    st.subheader("不同高度定常平飞性能")
    left, right = st.columns(2)
    with left:
        figure = px.line(
            valid_performance,
            x="speed_mps",
            y="thrust_N",
            color="height_m",
            markers=True,
            color_discrete_sequence=HEIGHT_COLORS,
            labels={"height_m": "高度 (m)"},
        )
        st.plotly_chart(
            style_plot(figure, "平飞需用推力", "真空速 V (m/s)", "需用推力 (N)"),
            use_container_width=True,
        )
    with right:
        figure = px.line(
            valid_performance,
            x="speed_mps",
            y="power_required_kW",
            color="height_m",
            markers=True,
            color_discrete_sequence=HEIGHT_COLORS,
            labels={"height_m": "高度 (m)"},
        )
        st.plotly_chart(
            style_plot(figure, "平飞气动需用功率", "真空速 V (m/s)", "需用功率 (kW)"),
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        figure = px.line(
            valid_performance,
            x="speed_mps",
            y="alpha_deg",
            color="height_m",
            markers=True,
            color_discrete_sequence=HEIGHT_COLORS,
            labels={"height_m": "高度 (m)"},
        )
        st.plotly_chart(
            style_plot(figure, "配平迎角", "真空速 V (m/s)", "迎角 α (°)"),
            use_container_width=True,
        )
    with right:
        figure = px.line(
            valid_performance,
            x="speed_mps",
            y="delta_e_deg",
            color="height_m",
            markers=True,
            color_discrete_sequence=HEIGHT_COLORS,
            labels={"height_m": "高度 (m)"},
        )
        figure.add_hline(y=0, line_width=1, line_color="#94A3B8")
        st.plotly_chart(
            style_plot(figure, "对称尾翼舵偏", "真空速 V (m/s)", "δe (°)"),
            use_container_width=True,
        )

    invalid_count = int((~performance["valid"]).sum())
    if invalid_count:
        st.warning(
            f"{invalid_count} 个配平点超出设定的气动数据有效范围，"
            "已从曲线中隐藏，但仍保留在明细表中。"
        )
    st.dataframe(performance, use_container_width=True, hide_index=True, height=380)

with tabs[2]:
    st.subheader("悬停构型与过渡飞行走廊")
    transition = results.get("transition")
    if not transition:
        st.info("本次未开启过渡走廊计算。可在左侧勾选“计算悬停构型与过渡走廊”后重新计算。")
    else:
        common_min, common_max = transition["common_speed_range"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("升力桨数量", transition["settings"]["propeller_count"])
        metric_cols[1].metric("桨径", f"{transition['settings']['propeller_diameter_m']:.2f} m")
        metric_cols[2].metric("倾角数量", len(transition["settings"]["tilt_angles_deg"]))
        if common_min is not None and common_max is not None and common_min <= common_max:
            metric_cols[3].metric("建议过渡速度带", f"{common_min:.0f}-{common_max:.0f} m/s")
        else:
            metric_cols[3].metric("建议过渡速度带", "无公共区间")

        st.caption(
            "beta=0 deg 为巡航推力方向，beta=90 deg 为悬停构型竖直推力方向。"
            "过渡段功率采用前向有效功率与垂向诱导功率估算。"
        )

        hover = transition_frame(results, "hover_rows")
        tilt = transition_frame(results, "tilt_rows")
        corridor = transition_frame(results, "corridor")
        valid_hover = hover[hover["valid"]].copy()
        valid_tilt = tilt[tilt["valid"]].copy()

        left, right = st.columns(2)
        with left:
            figure = px.line(
                valid_hover,
                x="speed_mps",
                y="thrust_total_N",
                color="height_m",
                markers=True,
                color_discrete_sequence=HEIGHT_COLORS,
                labels={"height_m": "高度 (m)"},
            )
            st.plotly_chart(
                style_plot(figure, "悬停构型平飞需用推力", "真空速 V (m/s)", "总需用推力 (N)"),
                use_container_width=True,
            )
        with right:
            figure = px.line(
                valid_hover,
                x="speed_mps",
                y="estimated_power_kW",
                color="height_m",
                markers=True,
                color_discrete_sequence=HEIGHT_COLORS,
                labels={"height_m": "高度 (m)"},
            )
            st.plotly_chart(
                style_plot(figure, "悬停构型功率估算", "真空速 V (m/s)", "估算功率 (kW)"),
                use_container_width=True,
            )

        left, right = st.columns(2)
        with left:
            figure = px.line(
                valid_tilt,
                x="speed_mps",
                y="thrust_total_N",
                color="tilt_deg",
                markers=True,
                labels={"tilt_deg": "倾角 beta (deg)"},
            )
            st.plotly_chart(
                style_plot(figure, "不同倾角需用推力", "真空速 V (m/s)", "总需用推力 (N)"),
                use_container_width=True,
            )
        with right:
            figure = px.line(
                valid_tilt,
                x="speed_mps",
                y="estimated_power_kW",
                color="tilt_deg",
                markers=True,
                labels={"tilt_deg": "倾角 beta (deg)"},
            )
            st.plotly_chart(
                style_plot(figure, "不同倾角功率估算", "真空速 V (m/s)", "估算功率 (kW)"),
                use_container_width=True,
            )

        corridor_figure = go.Figure()
        for _, row in corridor.dropna(subset=["min_speed_mps", "max_speed_mps"]).iterrows():
            corridor_figure.add_trace(
                go.Scatter(
                    x=[row["min_speed_mps"], row["max_speed_mps"]],
                    y=[row["tilt_deg"], row["tilt_deg"]],
                    mode="lines+markers+text",
                    text=["", f"{row['min_speed_mps']:.0f}-{row['max_speed_mps']:.0f}"],
                    textposition="middle right",
                    line=dict(width=8),
                    name=f"{row['tilt_deg']:.0f} deg",
                    showlegend=False,
                )
            )
        if common_min is not None and common_max is not None and common_min <= common_max:
            corridor_figure.add_vrect(
                x0=common_min,
                x1=common_max,
                fillcolor="#70AD47",
                opacity=0.16,
                line_width=0,
                annotation_text="共同过渡速度带",
                annotation_position="top left",
            )
        st.plotly_chart(
            style_plot(corridor_figure, "海平面过渡飞行走廊", "真空速 V (m/s)", "推力倾角 beta (deg)"),
            use_container_width=True,
        )

        st.markdown("**悬停构型平飞性能汇总**")
        st.dataframe(
            pd.DataFrame(transition["hover_summary"]).style.format(
                {
                    "min_thrust_N": "{:.1f}",
                    "max_thrust_N": "{:.1f}",
                    "min_power_kW": "{:.1f}",
                    "max_power_kW": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("**不同倾角有效速度区间**")
        st.dataframe(
            corridor.style.format(
                {
                    "min_speed_mps": "{:.0f}",
                    "max_speed_mps": "{:.0f}",
                    "min_thrust_N": "{:.1f}",
                    "max_thrust_N": "{:.1f}",
                    "min_power_kW": "{:.1f}",
                    "max_power_kW": "{:.1f}",
                },
                na_rep="-",
            ),
            use_container_width=True,
            hide_index=True,
        )

with tabs[3]:
    st.subheader("巡航模态特性")
    display_modal = modal.copy()
    display_modal["特征根"] = display_modal.apply(
        lambda row: (
            f"{row['real_1_s']:.5f} ± j{abs(row['imag_rad_s']):.5f}"
            if abs(row["imag_rad_s"]) > 1e-8
            else f"{row['real_1_s']:.5f}"
        ),
        axis=1,
    )
    st.dataframe(
        display_modal[
            [
                "height_m", "axis", "mode", "特征根",
                "natural_frequency_rad_s", "damping_ratio", "period_s",
                "half_or_double_time_s", "stable",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    figure = go.Figure()
    for height, color in zip(results["heights"], HEIGHT_COLORS):
        model = results["linear_models"][height]
        for axis, eigenvalues, symbol in [
            ("纵向", model["longitudinal_eigenvalues"], "circle"),
            ("横航向", model["lateral_eigenvalues"], "circle-open"),
        ]:
            figure.add_trace(
                go.Scatter(
                    x=[value.real for value in eigenvalues],
                    y=[value.imag for value in eigenvalues],
                    mode="markers",
                    name=f"{height} m · {axis}",
                    marker=dict(color=color, size=10, symbol=symbol),
                )
            )
    figure.add_vline(x=0, line_color="#C94C4C", line_dash="dash")
    figure.add_hline(y=0, line_color="#94A3B8")
    st.plotly_chart(
        style_plot(figure, "特征根复平面", "实部 σ (1/s)", "虚部 ω (rad/s)"),
        use_container_width=True,
    )

    oscillatory = modal[modal["mode"].isin(["长周期", "短周期", "荷兰滚"])]
    left, right = st.columns(2)
    with left:
        figure = px.line(
            oscillatory, x="height_m", y="damping_ratio", color="mode", markers=True
        )
        st.plotly_chart(
            style_plot(figure, "振荡模态阻尼比", "高度 h (m)", "阻尼比 ζ"),
            use_container_width=True,
        )
    with right:
        figure = px.line(
            oscillatory, x="height_m", y="period_s", color="mode", markers=True
        )
        st.plotly_chart(
            style_plot(figure, "振荡模态周期", "高度 h (m)", "周期 T (s)"),
            use_container_width=True,
        )

with tabs[4]:
    st.subheader("线化状态矩阵")
    matrix_height = st.selectbox("选择高度", results["heights"])
    model = results["linear_models"][matrix_height]
    left, right = st.columns(2)
    with left:
        st.markdown("**纵向 A 矩阵** — 状态 `[u, w, q, θ]`")
        st.dataframe(
            pd.DataFrame(
                model["longitudinal_A"],
                index=["u̇", "ẇ", "q̇", "θ̇"],
                columns=["u", "w", "q", "θ"],
            ).style.format("{:.6f}"),
            use_container_width=True,
        )
    with right:
        st.markdown("**横航向 A 矩阵** — 状态 `[v, p, r, φ]`")
        st.dataframe(
            pd.DataFrame(
                model["lateral_A"],
                index=["v̇", "ṗ", "ṙ", "φ̇"],
                columns=["v", "p", "r", "φ"],
            ).style.format("{:.6f}"),
            use_container_width=True,
        )
    st.markdown("**横航向导数（当前配平迎角）**")
    st.json(model["lateral_derivatives"])

with tabs[5]:
    st.subheader("下载结果")
    st.download_button(
        "下载本次计算结果（ZIP）",
        data=make_download_zip(results),
        file_name="evtol_analysis_results.zip",
        mime="application/zip",
        type="primary",
    )
    if REPORT_DOCX.exists():
        st.download_button(
            "下载巡航平飞与模态 Word 报告",
            data=REPORT_DOCX.read_bytes(),
            file_name=REPORT_DOCX.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    if TRANSITION_REPORT_DOCX.exists():
        st.download_button(
            "下载悬停构型与过渡走廊 Word 报告",
            data=TRANSITION_REPORT_DOCX.read_bytes(),
            file_name=TRANSITION_REPORT_DOCX.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    st.caption("ZIP 中包含巡航平飞 CSV、模态 CSV、过渡走廊 CSV 和完整 JSON 数据。")











