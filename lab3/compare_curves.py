import numpy as np, matplotlib.pyplot as plt
r_s = np.load("data/episode_returns_short.npy")
r_f = np.load("data/episode_returns_full.npy")
def sma(y, k): 
    import numpy as np
    if k>len(y): return y
    return np.convolve(y, np.ones(k)/k, mode="valid")
plt.figure()
plt.plot(sma(r_s, max(1, len(r_s)//20)), label="short (20k)")
plt.plot(sma(r_f, max(1, len(r_f)//100)), label="full (500k)")
plt.xlabel("Episode (smoothed index)"); plt.ylabel("Return")
plt.title("Learning Curves — Short vs Full")
plt.legend(); plt.tight_layout(); plt.savefig("figs/learning_compare.png", dpi=150)
