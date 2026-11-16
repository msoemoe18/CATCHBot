import numpy as np
from scipy.optimize import curve_fit


def _x_model(t, x0, vx):
    return x0 + vx * t


def _y_model(t, y0, vy, g):
    return y0 + vy * t + 0.5 * g * t**2


def fit_trajectory(times, xs, ys, g=None):
    """Fit x(t) as a straight line and y(t) as a parabola (pixel coordinates, y down).

    If g is given, gravity is fixed at that value; otherwise it is fitted too.
    Returns a dict with x0, vx, y0, vy, g and t0 (the time of the first point).
    """
    t = np.asarray(times, dtype=float)
    t0 = t[0]
    t = t - t0
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    (x0, vx), _ = curve_fit(_x_model, t, xs)
    if g is None:
        (y0, vy, g), _ = curve_fit(_y_model, t, ys, p0=[ys[0], 0.0, 1000.0])
    else:
        (y0, vy), _ = curve_fit(lambda tt, a, b: _y_model(tt, a, b, g), t, ys)
    return {"x0": x0, "vx": vx, "y0": y0, "vy": vy, "g": g, "t0": t0}


def landing_time(y0, vy, g, y_target):
    """Time after t=0 when the object crosses y_target while moving down. None if never."""
    a, b, c = 0.5 * g, vy, y0 - y_target
    if abs(a) < 1e-9:
        roots = [-c / b] if abs(b) > 1e-9 else []
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        root = np.sqrt(disc)
        roots = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
    falling = [r for r in roots if r > 0 and vy + g * r > 0]
    return min(falling) if falling else None


def predict_landing(fit, y_target):
    """Returns (x_land, t_land) where t_land is on the same clock as the input times."""
    t = landing_time(fit["y0"], fit["vy"], fit["g"], y_target)
    if t is None:
        return None
    return fit["x0"] + fit["vx"] * t, fit["t0"] + t


def landing_from_state(x, y, vx, vy, g, y_target):
    """Same prediction, but from a Kalman filter state instead of a curve fit."""
    t = landing_time(y, vy, g, y_target)
    if t is None:
        return None
    return x + vx * t, t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fps, noise_px = 60, 3.0
    true = {"x0": 100, "vx": 400, "y0": 300, "vy": -700, "g": 1500}
    y_land = 450

    t_true = landing_time(true["y0"], true["vy"], true["g"], y_land)
    x_true = true["x0"] + true["vx"] * t_true
    t = np.arange(0, t_true, 1 / fps)
    xs = true["x0"] + true["vx"] * t + rng.normal(0, noise_px, t.size)
    ys = true["y0"] + true["vy"] * t + 0.5 * true["g"] * t**2 + rng.normal(0, noise_px, t.size)

    print(f"True landing: x = {x_true:.1f} px at t = {t_true:.3f} s ({t.size} frames)\n")
    frames_used, errors = [], []
    for n in range(4, t.size + 1):
        result = predict_landing(fit_trajectory(t[:n], xs[:n], ys[:n]), y_land)
        if result is None:
            continue
        error = abs(result[0] - x_true)
        frames_used.append(n)
        errors.append(error)
        print(f"after {n:2d} frames ({t[n - 1]:.2f} s): x = {result[0]:6.1f} px, "
              f"error = {error:5.1f} px")

    final = fit_trajectory(t, xs, ys)
    tt = np.linspace(0, t_true, 100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.scatter(xs, ys, s=12, label="noisy detections")
    ax1.plot(_x_model(tt, final["x0"], final["vx"]),
             _y_model(tt, final["y0"], final["vy"], final["g"]), "r", label="fitted path")
    ax1.axhline(y_land, color="gray", linestyle="--", label="catch line")
    ax1.invert_yaxis()
    ax1.set(xlabel="x (px)", ylabel="y (px)", title="Simulated throw")
    ax1.legend()
    ax2.plot(frames_used, errors, "o-")
    ax2.set(xlabel="frames used", ylabel="landing error (px)",
            title="Prediction error vs. frames seen")
    plt.tight_layout()
    plt.savefig("landing_test.png", dpi=150)
    plt.show()