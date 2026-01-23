import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def spruce_budworm( t : float , x : float , r : float = 0.5 , k : float = 10)->float:
    """
    Spruce budworm model:
        dx/dt = r*x*(1 - x/k) - x^2/(1 + x^2)
    Parameters
    ----------
    t : float
        Time (not used, but required by ODE solvers).
    x : float
        Current budworm population.
    r : float
        Intrinsic growth rate.
    k : float
        Carrying capacity.
    Returns
    -------
    float
        Rate of change dx/dt.
    """
    dxdt = r * x * (1 - x / k) - (x**2) / (1 + x**2)
    return dxdt



def plot_spruce_budworm_rate(x_t: float, r: float = 0.5, k: float = 10.0) -> None:
    """
    Plots dx/dt as a function of x for the spruce budworm model.
    Shows equilibrium points and their stability.
    """

    # 1) Create population values from 0 to k
    x_values = np.linspace(0, k, 1000)

    # 2) Compute dx/dt for each x value
    dxdt_values = []
    for x in x_values:
        dxdt = spruce_budworm(0.0, x, r, k)
        dxdt_values.append(dxdt)

    # 3) Plot dx/dt vs x
    plt.figure()
    plt.plot(x_values, dxdt_values)
    plt.axhline(0)  # horizontal line at dx/dt = 0

    # 4) Find equilibrium points by checking sign changes
    for i in range(len(x_values) - 1):
        f_left = dxdt_values[i]
        f_right = dxdt_values[i + 1]

        # Check if dx/dt changes sign
        if f_left * f_right < 0:
            # Approximate equilibrium location
            x_star = (x_values[i] + x_values[i + 1]) / 2

            # 5) Determine stability
            if f_left > 0 and f_right < 0:
                # Stable equilibrium
                plt.plot(x_star, 0, "bo")  # blue dot
            elif f_left < 0 and f_right > 0:
                # Unstable equilibrium
                plt.plot(x_star, 0, "ro")  # red dot

    # 6) Mark current population
    plt.axvline(x_t, linestyle="--", color="green")

    # 7) Labels and title
    plt.xlabel("Population x")
    plt.ylabel("dx/dt")
    plt.title("Spruce Budworm Phase Portrait")

    plt.show()



def evolve_spruce_budworm(
    t: np.ndarray,
    x: np.ndarray,
    r: float = 0.5,
    k: float = 10.0,
    t_eval: float = 10.0,
    n_points: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evolve the spruce budworm system forward in time and append results.

    Parameters
    ----------
    t : np.ndarray
        Existing time array (must have at least one element).
    x : np.ndarray
        Existing population array (same length as t, must have at least one element).
    r : float, default=0.5
        Intrinsic growth rate.
    k : float, default=10.0
        Carrying capacity.
    t_eval : float, default=10.0
        Duration to evolve forward from the last time point.
    n_points : int, default=200
        Number of time points to evaluate inside the new interval.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Updated (t, x) arrays with the new solution appended.
    """

    # 1) Time span: start at last time, end at last time + duration
    t_start = t[-1]
    t_end = t[-1] + t_eval
    t_span = (t_start, t_end)

    # 2) Create evaluation points inside [t_start, t_end]
    t_eval_points = np.linspace(t_start, t_end, n_points)

    # 3) Solve the ODE starting from the last population value
    solution = solve_ivp(
        fun=spruce_budworm,
        t_span=t_span,
        y0=[x[-1]],
        t_eval=t_eval_points,
        args=(r, k),
        method="RK45"
    )

    # 4) Extract new results
    t_new = solution.t
    x_new = solution.y[0]

    # 5) Avoid duplicating the first point (it equals the last point of old arrays)
    #    We append from index 1 onward.
    t_appended = np.concatenate([t, t_new[1:]])
    x_appended = np.concatenate([x, x_new[1:]])

    # 6) Ensure population is never negative
    x_appended = np.clip(x_appended, 0.0, None)

    return t_appended, x_appended




def plot_spruce_budworm(t: np.ndarray, x: np.ndarray):
    """
    Plot the spruce budworm population over time.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        Population array.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes objects.
    """

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot population trajectory (green line)
    ax.plot(t, x, color="green")

    # Labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Population x(t)")
    ax.set_title("Spruce Budworm Population Over Time")

    # Ensure population axis starts at 0
    ax.set_ylim(bottom=0)

    # Add grid
    ax.grid(True)

    return fig, ax


# Initial condition
t = np.array([0.0])
x = np.array([0.5])

plot_spruce_budworm_rate(x_t=x[-1], r=0.7, k=10.0)


# Evolve the system forward in time
t, x = evolve_spruce_budworm(t, x, r=0.5, k=10.0, t_eval=50.0)

# NOW plot the time series
fig, ax = plot_spruce_budworm(t, x)
plt.show()

