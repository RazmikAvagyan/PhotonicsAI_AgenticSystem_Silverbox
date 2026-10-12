import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Simulated bench: laser -> DMD (amplitude only) -> scattering medium -> camera.
# The DMD has N x N micromirrors, the camera has M x M pixels. The medium is one
# fixed random complex transmission matrix T, so every experiment sees the same medium.
N = 32          # micromirrors per side
M = 32          # camera pixels per side
NOISE = 0.05    # relative noise of one camera shot
CENTER = M // 2  # the focus region is centred here
GAP = 1         # pixels between the focus region and the background

_rng = np.random.default_rng(1)
T = (_rng.normal(size=(M * M, N * N)) + 1j * _rng.normal(size=(M * M, N * N))) / np.sqrt(2 * N * N)


def camera(mask, rng):
    """One camera shot of a DMD pattern: |T Ein|^2 plus noise."""
    e_out = T @ mask.ravel()
    intensity = np.abs(e_out) ** 2
    return (intensity * (1 + NOISE * rng.normal(size=intensity.shape))).reshape(M, M)


def regions(focus_size, background_width, n_regions):
    """Boolean masks: the focus square (focus_size x focus_size pixels) and the background sectors."""
    y, x = np.mgrid[0:M, 0:M]
    low = CENTER - focus_size // 2                           # first row and column of the focus square
    high = low + focus_size - 1                              # last row and column of the focus square
    focus = (y >= low) & (y <= high) & (x >= low) & (x <= high)
    # square distance of every pixel from the focus square (0 inside it)
    dist = np.maximum(np.maximum(low - y, y - high).clip(0), np.maximum(low - x, x - high).clip(0))
    ring = (dist > GAP) & (dist <= GAP + background_width)   # background ring around the focus
    angle = np.arctan2(y - CENTER, x - CENTER) % (2 * np.pi)
    sector = (angle / (2 * np.pi) * n_regions).astype(int)   # split the ring into n_regions sectors
    return focus, [ring & (sector == i) for i in range(n_regions)]


def enhancement(image, focus, background):
    """Focus intensity divided by the mean of the background region means."""
    return image[focus].mean() / np.mean([image[r].mean() for r in background])


def run_focusing(macropixel, focus_size, background_width, n_regions, iterations, flips, seed, save_to):
    """Run one focusing search on the bench and return the enhancement and the saved image."""
    rng = np.random.default_rng(seed)                        # seed: random start pattern and camera noise
    side = N // macropixel                                   # macropixels per side
    pattern = rng.integers(0, 2, size=(side, side))          # random start pattern: each macropixel on (1) or off (0)
    focus, background = regions(focus_size, background_width, n_regions)  # focus square and background sectors

    def dmd(p):
        # A macropixel is a block of macropixel x macropixel micromirrors that switch together.
        return np.kron(p, np.ones((macropixel, macropixel)))  # macropixels -> micromirrors

    first_image = camera(dmd(pattern), rng)                  # camera image of the random pattern (speckle)
    current = enhancement(first_image, focus, background)    # enhancement of the random pattern
    start = current                                          # remembered for the report
    history = [current]                                      # enhancement after each iteration, for the plot

    for _ in range(iterations):
        trial = pattern.copy()                               # try a change on a copy of the pattern
        for k in rng.choice(side * side, size=min(flips, side * side), replace=False):
            trial[k // side, k % side] ^= 1                  # flip some macropixels (0 -> 1, 1 -> 0)
        value = enhancement(camera(dmd(trial), rng), focus, background)  # transmit, read focus and background
        if value > current:                                  # keep the flip only if it helped
            pattern, current = trial, value
        history.append(current)

    # Report a fresh shot of the final pattern (not the best of many noisy readings).
    last_image = camera(dmd(pattern), rng)
    final = enhancement(last_image, focus, background)

    # One figure: random-pattern image, focused image (same colour scale) and the enhancement curve.
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    top = last_image.max()
    for a, img, title in [(ax[0], first_image, f"random pattern, enhancement {start:.1f}"),
                          (ax[1], last_image, f"after search, enhancement {final:.1f}")]:
        a.imshow(img, cmap="inferno", vmin=0, vmax=top)
        a.contour(focus, levels=[0.5], colors="cyan", linewidths=0.8)   # outline of the focus region
        a.set_title(title, fontsize=10)
        a.axis("off")
    ax[2].plot(history)
    ax[2].set_xlabel("iteration")
    ax[2].set_ylabel("enhancement (running value)")
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_to, dpi=130)
    plt.close(fig)

    return {"enhancement": float(final), "start_enhancement": float(start),
            "macropixels": side * side, "image": str(save_to)}