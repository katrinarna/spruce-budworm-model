import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# -----------------------------
# Model: dx/dt
# -----------------------------
def spruce_budworm(t: float, x: float, r: float = 0.5, k: float = 10.0) -> float:
    """Spruce budworm model: dx/dt = r*x*(1-x/k) - x^2/(1+x^2)."""
    return r * x * (1 - x / k) - (x**2) / (1 + x**2)


# -----------------------------
# Phase portrait: dx/dt vs x
# (Stable blue, unstable red, current x_t green dashed line)
# -----------------------------
def plot_spruce_budworm_rate(x_t: float, r: float = 0.5, k: float = 10.0):
    """
    Plot dx/dt vs x for x in [0, k]. Mark equilibria:
      - stable (+ -> -) as blue circles
      - unstable (- -> +) as red circles
    Mark current population x_t as a green dashed line.
    Returns (fig, ax).
    """
    x_values = np.linspace(0, k, 3000)

    dxdt_values = []
    for x in x_values:
        dxdt_values.append(spruce_budworm(0.0, x, r, k))

    fig, ax = plt.subplots()
    ax.plot(x_values, dxdt_values)
    ax.axhline(0.0)

    # Find equilibria by sign changes (beginner method)
    for i in range(len(x_values) - 1):
        f_left = dxdt_values[i]
        f_right = dxdt_values[i + 1]

        if f_left * f_right < 0:
            x_star = (x_values[i] + x_values[i + 1]) / 2

            # Stable: + -> -
            if f_left > 0 and f_right < 0:
                ax.plot(x_star, 0, "bo", markersize=7)
            # Unstable: - -> +
            elif f_left < 0 and f_right > 0:
                ax.plot(x_star, 0, "ro", markersize=7)

    # current population
    ax.axvline(x_t, linestyle="--", color="green", linewidth=2)

    ax.set_xlim(0, k)
    ax.set_xlabel("Population x")
    ax.set_ylabel("dx/dt")
    ax.set_title("Phase Portrait (blue=stable, red=unstable)")
    ax.grid(True)

    return fig, ax


# -----------------------------
# Evolve forward (solve_ivp RK45)
# -----------------------------
def evolve_spruce_budworm(
    t: np.ndarray,
    x: np.ndarray,
    r: float = 0.5,
    k: float = 10.0,
    t_eval: float = 10.0,
    n_points: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evolve the system forward by duration t_eval starting from last (t[-1], x[-1]).
    Appends the new solution to existing arrays. Clamps x >= 0.
    """
    t_start = float(t[-1])
    t_end = t_start + float(t_eval)
    t_span = (t_start, t_end)

    # evaluation points in the interval
    t_eval_points = np.linspace(t_start, t_end, n_points)

    sol = solve_ivp(
        fun=spruce_budworm,
        t_span=t_span,
        y0=[float(x[-1])],
        t_eval=t_eval_points,
        args=(r, k),
        method="RK45"
    )

    t_new = sol.t
    x_new = sol.y[0]

    # append without duplicating the first point
    t_out = np.concatenate([t, t_new[1:]])
    x_out = np.concatenate([x, x_new[1:]])

    # population must be nonnegative
    x_out = np.clip(x_out, 0.0, None)

    return t_out, x_out


# -----------------------------
# Time series plot: x(t) vs t
# -----------------------------
def plot_spruce_budworm(t: np.ndarray, x: np.ndarray):
    """Plot x(t) over time in green. Returns (fig, ax)."""
    fig, ax = plt.subplots()
    ax.plot(t, x, color="green")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Population x(t)")
    ax.set_title("Population Over Time")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    return fig, ax


# =============================
# Streamlit App
# =============================
st.title("Spruce Budworm Population Dynamics")

# Sidebar controls
r = st.sidebar.slider("Intrinsic growth rate (r)", 0.0, 1.0, 0.5, 0.01)
k = st.sidebar.slider("Carrying capacity (k)", 0.1, 10.0, 10.0, 0.1)

# app default initial condition
x0 = k / 10.0

duration = st.sidebar.slider("Time to evolve (duration)", 1, 100, 10, 1)

col1, col2 = st.sidebar.columns(2)
evolve_btn = col1.button("Evolve Forward")
reset_btn = col2.button("Reset")

# Show the differential equation with current params
st.latex(rf"\frac{{dx}}{{dt}} = {r:.2f}x\left(1-\frac{{x}}{{{k:.2f}}}\right) - \frac{{x^2}}{{1+x^2}}")

# Session state initialization (or reset)
if reset_btn or ("sbw_t" not in st.session_state) or ("sbw_x" not in st.session_state):
    st.session_state["sbw_t"] = np.array([0.0])
    st.session_state["sbw_x"] = np.array([x0])

# If k changed a lot, x0 changes; optionally keep current x, but clamp within [0, k]
# (Beginner-friendly: just clip current x to [0, k])
st.session_state["sbw_x"] = np.clip(st.session_state["sbw_x"], 0.0, k)

# Retrieve current simulation arrays
t = st.session_state["sbw_t"]
x = st.session_state["sbw_x"]

# Evolve if button pressed
if evolve_btn:
    t, x = evolve_spruce_budworm(t, x, r=r, k=k, t_eval=duration)
    st.session_state["sbw_t"] = t
    st.session_state["sbw_x"] = x

# Plots
fig1, ax1 = plot_spruce_budworm_rate(x_t=float(x[-1]), r=r, k=k)
st.pyplot(fig1)

fig2, ax2 = plot_spruce_budworm(t, x)
st.pyplot(fig2)

# Small text to interpret direction
dxdt_now = spruce_budworm(0.0, float(x[-1]), r=r, k=k)
if dxdt_now > 0:
    st.write(f"At current xₜ = {x[-1]:.3f}, dx/dt = {dxdt_now:.4f} > 0 → population tends to increase.")
elif dxdt_now < 0:
    st.write(f"At current xₜ = {x[-1]:.3f}, dx/dt = {dxdt_now:.4f} < 0 → population tends to decrease.")
else:
    st.write(f"At current xₜ = {x[-1]:.3f}, dx/dt ≈ 0 → near an equilibrium.")
