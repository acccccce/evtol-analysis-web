"""
eVTOL 定常平飞性能与巡航模态特性分析
========================================

功能：
1. 从《eVTOL三视图与参数.docx》直接提取几何、惯量和气动数据；
2. 采用纵向三自由度非线性方程组求解定常平飞配平；
3. 计算不同高度的需用推力、气动需用功率曲线；
4. 在 65 m/s 巡航配平点数值线化，计算纵向和横航向模态；
5. 输出 JSON、CSV 和 PNG 图。

依赖：numpy, scipy, matplotlib, lxml
运行：python evtol_flight_dynamics_analysis.py
"""

from __future__ import annotations

import csv
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from lxml import etree
from matplotlib.font_manager import FontProperties
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.optimize import root


# -----------------------------------------------------------------------------
# 用户设置
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
SOURCE_DOCX = PROJECT_DIR / "eVTOL三视图与参数.docx"
OUTPUT_DIR = SCRIPT_DIR / "python_results"

HEIGHTS_M = [0, 1000, 2000, 3000]
SPEEDS_MPS = list(range(40, 111, 5))
MODAL_SPEED_MPS = 65.0

# 为避免气动/舵效数据过度外推而设置的结果有效范围。
VALID_ALPHA_RANGE_DEG = (-4.0, 12.0)
MAX_ABS_TAIL_DEFLECTION_DEG = 20.0

G = 9.80665


# -----------------------------------------------------------------------------
# 数据读取
# -----------------------------------------------------------------------------
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = {"w": WORD_NS, "m": MATH_NS}


def read_word_tables(path: Path) -> list[list[list[str]]]:
    """直接读取 DOCX 压缩包中的 document.xml，不依赖 python-docx。"""
    with zipfile.ZipFile(path, "r") as archive:
        root = etree.fromstring(archive.read("word/document.xml"))

    output = []
    for table in root.xpath(".//w:tbl", namespaces=XML_NS):
        rows = []
        for row in table.xpath("./w:tr", namespaces=XML_NS):
            cells = []
            for cell in row.xpath("./w:tc", namespaces=XML_NS):
                parts = cell.xpath(".//w:t/text() | .//m:t/text()", namespaces=XML_NS)
                cells.append("".join(parts).strip())
            rows.append(cells)
        output.append(rows)
    return output


def as_float(value: str) -> float:
    return float(value.strip())


# -----------------------------------------------------------------------------
# 飞机模型
# -----------------------------------------------------------------------------
class AircraftModel:
    def __init__(self, source_docx: Path):
        tables = read_word_tables(source_docx)

        geometry = {}
        for row in tables[1][1:]:
            geometry[row[0]] = as_float(row[1])
            geometry[row[2]] = as_float(row[3])

        self.mass = geometry["最大起飞质量/kg"]
        self.S = geometry["机翼参考面积/m2"]
        self.b = geometry["机翼展长/m"]
        self.c_bar = geometry["平均气动弦长/m"]
        self.length = geometry["全机长度/m"]
        self.Ixx = geometry["Ixx/(kg·m2)"]
        self.Iyy = geometry["Iyy/(kg·m2)"]
        self.Izz = geometry["Izz/(kg·m2)"]
        self.Ixz = geometry["Ixz/(kg·m2)"]
        self.weight = self.mass * G

        # 纵向静气动数据：alpha, CL, CD, Cm, CL/CD
        longitudinal = np.array(
            [[as_float(x) for x in row] for row in tables[4][1:]], dtype=float
        )
        alpha = longitudinal[:, 0]
        self.CL0 = PchipInterpolator(alpha, longitudinal[:, 1], extrapolate=True)
        self.CD0 = PchipInterpolator(alpha, longitudinal[:, 2], extrapolate=True)
        self.Cm0 = PchipInterpolator(alpha, longitudinal[:, 3], extrapolate=True)

        # 左右尾翼对称偏转：纵向效能相加。
        tail = np.array(
            [[as_float(x) for x in row] for row in tables[8][1:]], dtype=float
        )
        tail_alpha = tail[:, 0]
        self.CL_delta_e = PchipInterpolator(
            tail_alpha, 2.0 * tail[:, 1], extrapolate=True
        )
        self.CD_delta_e = PchipInterpolator(
            tail_alpha, 2.0 * tail[:, 2], extrapolate=True
        )
        self.Cm_delta_e = PchipInterpolator(
            tail_alpha, 2.0 * tail[:, 5], extrapolate=True
        )

        # 横航向静导数。原表中的 Cc 列按侧力系数 CY 使用。
        lateral_rows = []
        previous_alpha = None
        for row in tables[5][1:]:
            # Word 对 alpha 使用了纵向合并单元格；XML 中后续行为空，需向下填充。
            if row[0].strip():
                previous_alpha = row[0]
            else:
                row[0] = previous_alpha
            lateral_rows.append([as_float(x) for x in row])
        lateral = np.array(lateral_rows, dtype=float)
        alpha_nodes = sorted(set(lateral[:, 0]))
        slopes = {"CY": [], "Cl": [], "Cn": []}
        for alpha_i in alpha_nodes:
            rows = lateral[lateral[:, 0] == alpha_i]
            row_0 = rows[np.argmin(np.abs(rows[:, 1]))]
            row_4 = rows[np.argmin(np.abs(rows[:, 1] - 4.0))]
            slopes["CY"].append((row_4[4] - row_0[4]) / 4.0)
            slopes["Cl"].append((row_4[5] - row_0[5]) / 4.0)
            slopes["Cn"].append((row_4[7] - row_0[7]) / 4.0)

        self.lateral_slopes_per_deg = {
            key: PchipInterpolator(alpha_nodes, values, extrapolate=True)
            for key, values in slopes.items()
        }

        # 动导数：alpha, V, Cmq, CLq, Clp, Clr, Cnp, Cnr, CYp, CYr
        dynamic = np.array(
            [[as_float(x) for x in row] for row in tables[6][1:]], dtype=float
        )
        dynamic = dynamic[np.argsort(dynamic[:, 0])]
        names = ["alpha", "V", "Cmq", "CLq", "Clp", "Clr", "Cnp", "Cnr", "CYp", "CYr"]
        self.dynamic_derivatives = {
            name: interp1d(
                dynamic[:, 0],
                dynamic[:, column],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            for column, name in enumerate(names)
            if column >= 2
        }

    def longitudinal_coefficients(
        self, alpha_rad: float, delta_e_deg: float
    ) -> tuple[float, float, float]:
        alpha_deg = math.degrees(alpha_rad)
        cl = self.CL0(alpha_deg) + self.CL_delta_e(alpha_deg) * delta_e_deg
        cd = self.CD0(alpha_deg) + self.CD_delta_e(alpha_deg) * delta_e_deg
        cm = self.Cm0(alpha_deg) + self.Cm_delta_e(alpha_deg) * delta_e_deg
        return float(cl), float(cd), float(cm)


# -----------------------------------------------------------------------------
# 大气与非线性配平
# -----------------------------------------------------------------------------
def isa_density(height_m: float) -> float:
    """ISA 对流层密度，适用于本项目 0~3000 m 高度范围。"""
    temperature_0 = 288.15
    pressure_0 = 101325.0
    lapse_rate = 0.0065
    gas_constant = 287.05287

    temperature = temperature_0 - lapse_rate * height_m
    pressure = pressure_0 * (temperature / temperature_0) ** (
        G / (gas_constant * lapse_rate)
    )
    return pressure / (gas_constant * temperature)


def solve_level_trim(
    aircraft: AircraftModel,
    height_m: float,
    speed_mps: float,
    initial_guess: np.ndarray | None = None,
) -> dict:
    """
    求解未知量 [alpha(rad), delta_e(deg), thrust(N)]。

    定常平飞条件：gamma=0, theta=alpha, q=0。
    方程：
        X/m - g sin(theta) = 0
        Z/m + g cos(theta) = 0
        Cm = 0
    """
    rho = isa_density(height_m)
    dynamic_pressure = 0.5 * rho * speed_mps**2

    if initial_guess is None:
        required_cl = aircraft.weight / (dynamic_pressure * aircraft.S)
        alpha_grid = np.linspace(-4.0, 12.0, 321)
        alpha_deg = float(
            alpha_grid[np.argmin(np.abs(aircraft.CL0(alpha_grid) - required_cl))]
        )
        cm_delta = float(aircraft.Cm_delta_e(alpha_deg))
        delta_e = -float(aircraft.Cm0(alpha_deg)) / cm_delta
        cd = float(
            aircraft.CD0(alpha_deg) + aircraft.CD_delta_e(alpha_deg) * delta_e
        )
        thrust = max(1.0, dynamic_pressure * aircraft.S * cd)
        initial_guess = np.array([math.radians(alpha_deg), delta_e, thrust])

    def residual(unknowns: np.ndarray) -> np.ndarray:
        alpha, delta_e, thrust = unknowns
        cl, cd, cm = aircraft.longitudinal_coefficients(alpha, delta_e)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)

        force_x = (
            dynamic_pressure
            * aircraft.S
            * (-cd * cos_alpha + cl * sin_alpha)
            + thrust
        )
        force_z = (
            dynamic_pressure
            * aircraft.S
            * (-cd * sin_alpha - cl * cos_alpha)
        )
        return np.array(
            [
                force_x / aircraft.mass - G * sin_alpha,
                force_z / aircraft.mass + G * cos_alpha,
                cm,
            ]
        )

    solution = root(residual, initial_guess, method="hybr", tol=1e-11)
    alpha, delta_e, thrust = solution.x
    cl, cd, cm = aircraft.longitudinal_coefficients(alpha, delta_e)
    lift = dynamic_pressure * aircraft.S * cl
    drag = dynamic_pressure * aircraft.S * cd

    valid = bool(
        solution.success
        and VALID_ALPHA_RANGE_DEG[0]
        <= math.degrees(alpha)
        <= VALID_ALPHA_RANGE_DEG[1]
        and abs(delta_e) <= MAX_ABS_TAIL_DEFLECTION_DEG
        and thrust >= 0.0
        and cd > 0.0
    )

    # 严格平衡：L + T sin(alpha) = W，T cos(alpha) = D。
    return {
        "height_m": height_m,
        "speed_mps": speed_mps,
        "rho_kg_m3": rho,
        "alpha_deg": math.degrees(alpha),
        "delta_e_deg": float(delta_e),
        "thrust_N": float(thrust),
        "power_required_kW": float(drag * speed_mps / 1000.0),
        "CL": cl,
        "CD": cd,
        "Cm": cm,
        "lift_N": lift,
        "drag_N": drag,
        "lift_weight_ratio": lift / aircraft.weight,
        "thrust_drag_ratio": thrust / drag,
        "vertical_balance_ratio": (
            lift + thrust * math.sin(alpha)
        ) / aircraft.weight,
        "horizontal_balance_ratio": thrust * math.cos(alpha) / drag,
        "residual_norm": float(np.linalg.norm(residual(solution.x))),
        "solver_success": bool(solution.success),
        "valid": valid,
        "solution_vector": solution.x,
    }


# -----------------------------------------------------------------------------
# 线化与模态
# -----------------------------------------------------------------------------
def numerical_jacobian(function, state: np.ndarray, steps: list[float]) -> np.ndarray:
    jacobian = np.zeros((len(state), len(state)))
    for column, step in enumerate(steps):
        perturbation = np.zeros(len(state))
        perturbation[column] = step
        jacobian[:, column] = (
            function(state + perturbation) - function(state - perturbation)
        ) / (2.0 * step)
    return jacobian


def eigenvalue_metrics(eigenvalue: complex) -> dict:
    sigma = float(eigenvalue.real)
    omega = abs(float(eigenvalue.imag))
    omega_n = math.hypot(sigma, omega)
    return {
        "real_1_s": sigma,
        "imag_rad_s": float(eigenvalue.imag),
        "natural_frequency_rad_s": omega_n,
        "damping_ratio": -sigma / omega_n if omega > 1e-9 else None,
        "period_s": 2.0 * math.pi / omega if omega > 1e-9 else None,
        "half_or_double_time_s": (
            math.log(2.0) / abs(sigma) if abs(sigma) > 1e-12 else None
        ),
        "stable": sigma < 0.0,
    }


def calculate_longitudinal_modes(aircraft: AircraftModel, trim: dict) -> dict:
    speed = trim["speed_mps"]
    alpha_0 = math.radians(trim["alpha_deg"])
    u_0 = speed * math.cos(alpha_0)
    w_0 = speed * math.sin(alpha_0)
    rho = trim["rho_kg_m3"]
    delta_e = trim["delta_e_deg"]
    thrust = trim["thrust_N"]

    def state_equations(state: np.ndarray) -> np.ndarray:
        u, w, pitch_rate, theta = state
        local_speed = max(1e-8, math.hypot(u, w))
        alpha = math.atan2(w, u)
        alpha_deg = math.degrees(alpha)
        q_bar = 0.5 * rho * local_speed**2

        cl, cd, cm = aircraft.longitudinal_coefficients(alpha, delta_e)
        nondimensional_q = pitch_rate * aircraft.c_bar / (2.0 * local_speed)
        cl += float(aircraft.dynamic_derivatives["CLq"](alpha_deg)) * nondimensional_q
        cm += float(aircraft.dynamic_derivatives["Cmq"](alpha_deg)) * nondimensional_q

        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)
        force_x = q_bar * aircraft.S * (-cd * cos_alpha + cl * sin_alpha) + thrust
        force_z = q_bar * aircraft.S * (-cd * sin_alpha - cl * cos_alpha)
        moment_y = q_bar * aircraft.S * aircraft.c_bar * cm

        return np.array(
            [
                -pitch_rate * w + force_x / aircraft.mass - G * math.sin(theta),
                pitch_rate * u + force_z / aircraft.mass + G * math.cos(theta),
                moment_y / aircraft.Iyy,
                pitch_rate,
            ]
        )

    state_0 = np.array([u_0, w_0, 0.0, alpha_0])
    matrix_a = numerical_jacobian(
        state_equations,
        state_0,
        [speed * 1e-5, speed * 1e-5, 1e-6, 1e-6],
    )
    eigenvalues = np.linalg.eigvals(matrix_a)
    positive_imaginary = sorted(
        [value for value in eigenvalues if value.imag > 1e-7],
        key=lambda value: abs(value.imag),
    )
    modes = [
        {"mode": "长周期", **eigenvalue_metrics(positive_imaginary[0])},
        {"mode": "短周期", **eigenvalue_metrics(positive_imaginary[-1])},
    ]
    return {"A": matrix_a, "eigenvalues": eigenvalues, "modes": modes}


def calculate_lateral_modes(aircraft: AircraftModel, trim: dict) -> dict:
    speed = trim["speed_mps"]
    alpha_0 = math.radians(trim["alpha_deg"])
    alpha_deg = trim["alpha_deg"]
    u_0 = speed * math.cos(alpha_0)
    w_0 = speed * math.sin(alpha_0)
    rho = trim["rho_kg_m3"]

    # 原数据是每度导数，状态方程中的 beta 使用弧度，故乘 180/pi。
    cy_beta = float(aircraft.lateral_slopes_per_deg["CY"](alpha_deg)) * 180.0 / math.pi
    cl_beta = float(aircraft.lateral_slopes_per_deg["Cl"](alpha_deg)) * 180.0 / math.pi
    cn_beta = float(aircraft.lateral_slopes_per_deg["Cn"](alpha_deg)) * 180.0 / math.pi
    derivatives = {
        name: float(aircraft.dynamic_derivatives[name](alpha_deg))
        for name in ["Clp", "Clr", "Cnp", "Cnr", "CYp", "CYr"]
    }

    def state_equations(state: np.ndarray) -> np.ndarray:
        lateral_velocity, roll_rate, yaw_rate, phi = state
        local_speed = max(
            1e-8,
            math.sqrt(u_0**2 + w_0**2 + lateral_velocity**2),
        )
        beta = math.asin(np.clip(lateral_velocity / local_speed, -1.0, 1.0))
        q_bar = 0.5 * rho * local_speed**2
        p_hat = roll_rate * aircraft.b / (2.0 * local_speed)
        r_hat = yaw_rate * aircraft.b / (2.0 * local_speed)

        cy = cy_beta * beta + derivatives["CYp"] * p_hat + derivatives["CYr"] * r_hat
        cl = cl_beta * beta + derivatives["Clp"] * p_hat + derivatives["Clr"] * r_hat
        cn = cn_beta * beta + derivatives["Cnp"] * p_hat + derivatives["Cnr"] * r_hat

        force_y = q_bar * aircraft.S * cy
        moment_x = q_bar * aircraft.S * aircraft.b * cl
        moment_z = q_bar * aircraft.S * aircraft.b * cn

        inertia_determinant = aircraft.Ixx * aircraft.Izz - aircraft.Ixz**2
        roll_acceleration = (
            aircraft.Izz * moment_x + aircraft.Ixz * moment_z
        ) / inertia_determinant
        yaw_acceleration = (
            aircraft.Ixz * moment_x + aircraft.Ixx * moment_z
        ) / inertia_determinant

        return np.array(
            [
                roll_rate * w_0
                - yaw_rate * u_0
                + force_y / aircraft.mass
                + G * math.cos(alpha_0) * math.sin(phi),
                roll_acceleration,
                yaw_acceleration,
                roll_rate + math.tan(alpha_0) * yaw_rate,
            ]
        )

    matrix_a = numerical_jacobian(
        state_equations,
        np.zeros(4),
        [1e-4, 1e-6, 1e-6, 1e-6],
    )
    eigenvalues = np.linalg.eigvals(matrix_a)
    complex_root = max(
        [value for value in eigenvalues if value.imag > 1e-7],
        key=lambda value: abs(value.imag),
    )
    real_roots = sorted(
        [value for value in eigenvalues if abs(value.imag) <= 1e-7],
        key=lambda value: value.real,
    )
    modes = [
        {"mode": "荷兰滚", **eigenvalue_metrics(complex_root)},
        {"mode": "滚转收敛", **eigenvalue_metrics(real_roots[0])},
        {"mode": "螺旋模态", **eigenvalue_metrics(real_roots[-1])},
    ]
    return {
        "A": matrix_a,
        "eigenvalues": eigenvalues,
        "modes": modes,
        "derivatives": {
            "CY_beta_per_rad": cy_beta,
            "Cl_beta_per_rad": cl_beta,
            "Cn_beta_per_rad": cn_beta,
            **derivatives,
        },
    }


# -----------------------------------------------------------------------------
# 输出
# -----------------------------------------------------------------------------
def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    raise TypeError(f"无法序列化：{type(value)}")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def configure_chinese_plot_font() -> FontProperties | None:
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            font = FontProperties(fname=str(candidate))
            plt.rcParams["axes.unicode_minus"] = False
            return font
    return None


def make_plots(performance: list[dict], modal_rows: list[dict]) -> None:
    font = configure_chinese_plot_font()
    colors = ["#17365D", "#2E75B6", "#70AD47", "#C55A11"]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for height, color in zip(HEIGHTS_M, colors):
        rows = [
            row for row in performance
            if row["height_m"] == height and row["valid"]
        ]
        speed = [row["speed_mps"] for row in rows]
        axes[0].plot(speed, [row["thrust_N"] for row in rows], "-o", ms=3,
                     color=color, label=f"{height} m")
        axes[1].plot(speed, [row["power_required_kW"] for row in rows], "-o", ms=3,
                     color=color, label=f"{height} m")

    labels = [("需用推力 (N)", "平飞需用推力"),
              ("需用功率 (kW)", "平飞气动需用功率")]
    for axis, (ylabel, title) in zip(axes, labels):
        axis.set_xlabel("真空速 V (m/s)", fontproperties=font)
        axis.set_ylabel(ylabel, fontproperties=font)
        axis.set_title(title, fontproperties=font)
        axis.grid(alpha=0.25)
        axis.legend(prop=font)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "performance_curves.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode, marker, color in zip(
        ["长周期", "短周期", "荷兰滚"],
        ["o-", "s-", "^-"],
        colors[:3],
    ):
        rows = [row for row in modal_rows if row["mode"] == mode]
        axes[0].plot(
            [row["height_m"] for row in rows],
            [row["damping_ratio"] for row in rows],
            marker,
            color=color,
            label=mode,
        )
        axes[1].plot(
            [row["height_m"] for row in rows],
            [row["period_s"] for row in rows],
            marker,
            color=color,
            label=mode,
        )
    for axis, ylabel in zip(axes, ["阻尼比 ζ", "周期 T (s)"]):
        axis.set_xlabel("高度 h (m)", fontproperties=font)
        axis.set_ylabel(ylabel, fontproperties=font)
        axis.grid(alpha=0.25)
        axis.legend(prop=font)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "modal_characteristics.png", dpi=220)
    plt.close(figure)


def main() -> None:
    if not SOURCE_DOCX.exists():
        raise FileNotFoundError(f"找不到原始数据文件：{SOURCE_DOCX}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aircraft = AircraftModel(SOURCE_DOCX)

    performance = []
    for height in HEIGHTS_M:
        guess = None
        for speed in SPEEDS_MPS:
            trim = solve_level_trim(aircraft, height, speed, guess)
            guess = trim.pop("solution_vector")
            performance.append(trim)

    modal_rows = []
    linear_models = []
    for height in HEIGHTS_M:
        trim = solve_level_trim(aircraft, height, MODAL_SPEED_MPS)
        trim.pop("solution_vector")
        longitudinal = calculate_longitudinal_modes(aircraft, trim)
        lateral = calculate_lateral_modes(aircraft, trim)

        for mode in longitudinal["modes"]:
            modal_rows.append(
                {
                    "height_m": height,
                    "speed_mps": MODAL_SPEED_MPS,
                    "axis": "纵向",
                    **mode,
                }
            )
        for mode in lateral["modes"]:
            modal_rows.append(
                {
                    "height_m": height,
                    "speed_mps": MODAL_SPEED_MPS,
                    "axis": "横航向",
                    **mode,
                }
            )

        linear_models.append(
            {
                "height_m": height,
                "trim": trim,
                "longitudinal_A": longitudinal["A"],
                "longitudinal_eigenvalues": longitudinal["eigenvalues"],
                "lateral_A": lateral["A"],
                "lateral_eigenvalues": lateral["eigenvalues"],
                "lateral_derivatives": lateral["derivatives"],
            }
        )

    results = {
        "aircraft": {
            "mass_kg": aircraft.mass,
            "S_m2": aircraft.S,
            "b_m": aircraft.b,
            "c_bar_m": aircraft.c_bar,
            "Ixx_kg_m2": aircraft.Ixx,
            "Iyy_kg_m2": aircraft.Iyy,
            "Izz_kg_m2": aircraft.Izz,
            "Ixz_kg_m2": aircraft.Ixz,
        },
        "assumptions": {
            "heights_m": HEIGHTS_M,
            "speeds_mps": SPEEDS_MPS,
            "modal_speed_mps": MODAL_SPEED_MPS,
            "thrust_axis": "body x-axis through center of gravity",
            "power_definition": "D*V = T*V*cos(alpha)",
            "lateral_Cc_interpretation": "CY",
        },
        "performance": performance,
        "modal_results": modal_rows,
        "linear_models": linear_models,
    }
    with (OUTPUT_DIR / "analysis_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2, default=json_safe)

    performance_fields = [
        "height_m", "speed_mps", "rho_kg_m3", "valid",
        "alpha_deg", "delta_e_deg", "thrust_N", "power_required_kW",
        "CL", "CD", "lift_N", "drag_N", "lift_weight_ratio",
        "thrust_drag_ratio", "vertical_balance_ratio",
        "horizontal_balance_ratio", "residual_norm",
    ]
    write_csv(OUTPUT_DIR / "level_flight_performance.csv", performance, performance_fields)

    modal_fields = [
        "height_m", "speed_mps", "axis", "mode", "real_1_s",
        "imag_rad_s", "natural_frequency_rad_s", "damping_ratio",
        "period_s", "half_or_double_time_s", "stable",
    ]
    write_csv(OUTPUT_DIR / "modal_results.csv", modal_rows, modal_fields)
    make_plots(performance, modal_rows)

    print(f"分析完成，结果目录：{OUTPUT_DIR}")
    for model in linear_models:
        trim = model["trim"]
        print(
            f"h={model['height_m']:4.0f} m, V={MODAL_SPEED_MPS:.0f} m/s: "
            f"alpha={trim['alpha_deg']:.3f} deg, "
            f"delta_e={trim['delta_e_deg']:.3f} deg, "
            f"T={trim['thrust_N']:.1f} N"
        )
    for row in modal_rows:
        sign = "+" if row["imag_rad_s"] >= 0 else "-"
        print(
            f"{row['height_m']:4.0f} m {row['axis']} {row['mode']}: "
            f"lambda={row['real_1_s']:.6f} {sign} "
            f"j{abs(row['imag_rad_s']):.6f}, stable={row['stable']}"
        )


if __name__ == "__main__":
    main()




