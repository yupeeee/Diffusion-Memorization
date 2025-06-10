import os

import matplotlib.pyplot as plt
import utils
from paths import FIG_DIR

T = 999
pipe = utils.pipe.StableDiffusion(
    version="1.4",
    scheduler="DDIM",
    variant="fp16",
    verbose=False,
)
alphas = pipe.alphas_cumprod(T)

# Pre-compute sqrt terms
sqrt_alpha = alphas.sqrt()
sqrt_1_minus_alpha = (1 - alphas).sqrt()
sqrt_alpha_prev = alphas[1:].sqrt()
sqrt_1_minus_alpha_prev = (1 - alphas[1:]).sqrt()

# Calculate coefficients
image_coefs = (
    sqrt_alpha_prev * sqrt_1_minus_alpha[:-1]
    - sqrt_alpha[:-1] * sqrt_1_minus_alpha_prev
) / sqrt_1_minus_alpha[:-1]
noise_coefs = (sqrt_1_minus_alpha[:-1] - sqrt_1_minus_alpha_prev) / sqrt_1_minus_alpha[
    :-1
]
coefs = list(zip(image_coefs.tolist(), noise_coefs.tolist()))

# Adjust coefficients for x_0 and x_T
coefs[0] = (coefs[0][0], 1 - coefs[0][1])
for i in range(len(coefs) - 1):
    coefs[i + 1] = (
        coefs[i + 1][0] + (1 - coefs[i + 1][1]) * coefs[i][0],
        (1 - coefs[i + 1][1]) * coefs[i][1],
    )

image_coefs = [c[0] for c in coefs]
noise_coefs = [c[1] for c in coefs]

# Plot
plt.rcParams["font.size"] = 15
os.makedirs(FIG_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(3, 3))
ax.fill_between(
    range(len(coefs)),
    image_coefs,
    alpha=0.5,
    color="blue",
    label="$\\omega_{{0}}^{{(t)}}$",
)
ax.fill_between(
    range(len(coefs)),
    image_coefs,
    [sum(x) for x in zip(image_coefs, noise_coefs)],
    alpha=0.5,
    color="red",
    label="$\\omega_{{T}}^{{(t)}}$",
)
ax.plot(
    range(len(sqrt_alpha_prev)),
    sqrt_alpha_prev,
    "--",
    color="blue",
    label="$\\sqrt{{\\bar{{\\alpha}}_{{t-1}}}}$",
)
ax.plot(
    range(len(sqrt_alpha_prev)),
    sqrt_alpha_prev + sqrt_1_minus_alpha_prev,
    "--",
    color="red",
    label="$\\sqrt{{1-\\bar{{\\alpha}}_{{t-1}}}}$",
)
ax.set_xlabel("$t$")
ax.set_xticks([0, T], ["$T$", "$0$"])
ax.set_ylabel("Portion@$x_{{t}}$")
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.25)
plt.savefig(
    os.path.join(FIG_DIR, "coefs.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.05
)
plt.close("all")
