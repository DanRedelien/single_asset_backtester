import numpy as np

# Simulate Kalman
y = np.full(100, 8.5)
x = np.full(100, 9.5)

a, b = 0.0, 1.0
P00, P01, P10, P11 = 1.0, 0.0, 0.0, 1.0
Q, R = 1e-4, 1e-3

for t in range(5):
    xt = x[t]
    yt = y[t]
    P00 += Q
    P11 += Q
    S = P00 + xt*(P10 + P01) + xt*xt*P11 + R
    K0 = (P00 + P01*xt) / S
    K1 = (P10 + P11*xt) / S
    error = yt - (a + b * xt)
    print(f"t={t} a={a:.4f} b={b:.4f} error={error:.4f} P11={P11:.2e} S={S:.2f}")
    
    a = a + K0 * error
    b = b + K1 * error
    
    t00 = P00 - (K0*P00 + K0*xt*P10)
    t01 = P01 - (K0*P01 + K0*xt*P11)
    t10 = P10 - (K1*P00 + K1*xt*P10)
    t11 = P11 - (K1*P01 + K1*xt*P11)
    P00, P01, P10, P11 = t00, t01, t10, t11
