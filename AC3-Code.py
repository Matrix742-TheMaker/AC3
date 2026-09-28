# =========================
# AP CHEM KINETICS SIMULATOR
# One-click: press ▶ once
# =========================

import numpy as np
import matplotlib.pyplot as plt

# ---------- STUDENT INPUTS (edit only these 4 lines) ----------
A0 = 0.67        # initial concentration (M)
k = 1.00          # rate constant (units depend on order)
order = 0         # 0, 1, or 2
t_end = 0.66      # total time (arbitrary units)
# ------------------------------------------------------------

# Time axis
t = np.linspace(0, t_end, 200)

# Integrated rate law models
def A_zero(t, A0, k):   return A0 - k*t
def A_first(t, A0, k):  return A0 * np.exp(-k*t)
def A_second(t, A0, k): return 1 / (1/A0 + k*t)

# Generate [A](t)
if order == 0:
    A = A_zero(t, A0, k)
elif order == 1:
    A = A_first(t, A0, k)
elif order == 2:
    A = A_second(t, A0, k)
else:
    raise ValueError("order must be 0, 1, or 2")

# Safety for logs/division
A = np.clip(A, 1e-12, None)

# Helper: best-fit line and R^2
def best_fit(x, y):
    m, b = np.polyfit(x, y, 1)
    yhat = m*x + b
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot
    return m, b, r2

# Compute linearizations
m0, b0, r2_0 = best_fit(t, A)           # zero-order test
m1, b1, r2_1 = best_fit(t, np.log(A))   # first-order test
m2, b2, r2_2 = best_fit(t, 1/A)         # second-order test

# Graphs
plt.figure(figsize=(5,4))
plt.plot(t, A)
plt.xlabel("time")
plt.ylabel("[A] (M)")
plt.title("[A] vs time")
plt.show()

plt.figure(figsize=(5,4))
plt.plot(t, np.log(A))
plt.xlabel("time")
plt.ylabel("ln[A]")
plt.title("ln[A] vs time")
plt.show()

plt.figure(figsize=(5,4))
plt.plot(t, 1/A)
plt.xlabel("time")
plt.ylabel("1/[A]")
plt.title("1/[A] vs time")
plt.show()

# Report
print("Linearity check (higher R^2 = more linear):")
print(f"  [A] vs t:     R^2 = {r2_0:.5f}   (zero-order test)")
print(f"  ln[A] vs t:   R^2 = {r2_1:.5f}   (first-order test)")
print(f"  1/[A] vs t:   R^2 = {r2_2:.5f}   (second-order test)\n")

print("If the plot is linear, slope relates to k like this:")
print(f"  Zero order:   [A] = -kt + [A]0   => k = {-m0:.5f}")
print(f"  First order:  ln[A] = -kt + ln[A]0 => k = {-m1:.5f}")
print(f"  Second order: 1/[A] = kt + 1/[A]0 => k = {m2:.5f}\n")

print("You entered:")
print(f"  order = {order}, A0 = {A0}, k = {k}, t_end = {t_end}")