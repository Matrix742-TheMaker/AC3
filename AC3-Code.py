# =========================
# AP CHEM KINETICS SIMULATOR
# With Dynamic pH + pOH
# =========================

import numpy as np
import matplotlib.pyplot as plt

# ---------- STUDENT INPUTS ----------
A0 = 5          # initial concentration of reactant A (M)
k = 1           # rate constant
order = 1       # reaction order (0,1,2)
t_end = 10      # total time
stoich_A = 1    # coefficient of reactant
stoich_P = 2    # coefficient of product
H0 = 1e-7       # initial [H+] concentration
acidic_products = True   # True if reaction produces acid
# ------------------------------------

# Time axis
t = np.linspace(0, t_end, 300)

# ----------------------------
# Integrated Rate Law Models
# ----------------------------
def A_zero(t, A0, k):
    return A0 - k*t

def A_first(t, A0, k):
    return A0 * np.exp(-k*t)

def A_second(t, A0, k):
    return 1 / (1/A0 + k*t)

# Generate [A](t)
if order == 0:
    A = A_zero(t, A0, k)
elif order == 1:
    A = A_first(t, A0, k)
elif order == 2:
    A = A_second(t, A0, k)
else:
    raise ValueError("order must be 0,1,2")

A = np.clip(A, 1e-12, None)

# ----------------------------
# Product Formation
# ----------------------------
def product_concentration(A0, A_t, stoich_A, stoich_P):
    reacted = A0 - A_t
    return reacted * (stoich_P / stoich_A)

P = product_concentration(A0, A, stoich_A, stoich_P)

# ----------------------------
# Dynamic pH Model
# ----------------------------
if acidic_products:
    H = H0 + P
else:
    H = H0 - P
    H = np.clip(H, 1e-12, None)

OH = 1e-14 / H

pH = -np.log10(H)
pOH = -np.log10(OH)

# ----------------------------
# Best Fit Linearity Test
# ----------------------------
def best_fit(x, y):
    m, b = np.polyfit(x, y, 1)
    yhat = m*x + b
    ss_res = np.sum((y-yhat)**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot
    return m,b,r2

m0,b0,r2_0 = best_fit(t,A)
m1,b1,r2_1 = best_fit(t,np.log(A))
m2,b2,r2_2 = best_fit(t,1/A)

# ----------------------------
# Graphs
# ----------------------------

plt.figure()
plt.plot(t,A)
plt.xlabel("Time")
plt.ylabel("[A]")
plt.title("[A] vs Time")
plt.show()

plt.figure()
plt.plot(t,np.log(A))
plt.xlabel("Time")
plt.ylabel("ln[A]")
plt.title("ln[A] vs Time")
plt.show()

plt.figure()
plt.plot(t,1/A)
plt.xlabel("Time")
plt.ylabel("1/[A]")
plt.title("1/[A] vs Time")
plt.show()

# Reactant/Product

plt.figure()
plt.plot(t,A,label="Reactant [A]")
plt.plot(t,P,label="Product [P]")
plt.xlabel("Time")
plt.ylabel("Concentration (M)")
plt.title("Reaction Progress")
plt.legend()
plt.grid()
plt.show()

# pH and pOH

plt.figure()
plt.plot(t,pH,label="pH")
plt.plot(t,pOH,label="pOH")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("pH and pOH vs Time")
plt.legend()
plt.grid()
plt.show()

# ----------------------------
# Half Life Model
# ----------------------------
def model_half_life(A0,k,order):

    if order == 0:
        half_life = A0/(2*k)
        t_plot = np.linspace(0,A0/k,100)
        conc = A0 - k*t_plot

    elif order == 1:
        half_life = np.log(2)/k
        t_plot = np.linspace(0,half_life*4,100)
        conc = A0*np.exp(-k*t_plot)

    elif order == 2:
        half_life = 1/(k*A0)
        t_plot = np.linspace(0,half_life*4,100)
        conc = 1/(1/A0 + k*t_plot)

    else:
        raise ValueError("Order must be 0,1,2")

    plt.figure()
    plt.plot(t_plot,conc)
    plt.axvline(half_life,linestyle="--")
    plt.axhline(A0/2,linestyle="--")
    plt.title(f"{order}-Order Half Life")
    plt.xlabel("Time")
    plt.ylabel("[A]")
    plt.grid()
    plt.show()

    return half_life

t_half = model_half_life(A0,k,order)

# ----------------------------
# Results
# ----------------------------
print("Linearity check (higher R^2 = more linear):")
print(f"[A] vs t R^2 = {r2_0:.5f}")
print(f"ln[A] vs t R^2 = {r2_1:.5f}")
print(f"1/[A] vs t R^2 = {r2_2:.5f}")

print("\nRate constant if linear:")
print(f"Zero order k = {-m0:.5f}")
print(f"First order k = {-m1:.5f}")
print(f"Second order k = {m2:.5f}")

print("\nHalf Life =", t_half)

print("\nFinal pH =", pH[-1])
print("Final pOH =", pOH[-1])