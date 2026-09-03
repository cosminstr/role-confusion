import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def compare_act(x: torch.Tensor, y: torch.Tensor):
    cosine_similarity = F.cosine_similarity(x, y, dim=-1)
    euclidean_distance = torch.linalg.vector_norm(x - y, dim=-1)

    normalized_euclidean = euclidean_distance / (
        torch.linalg.vector_norm(x, dim=-1) + torch.linalg.vector_norm(y, dim=-1) + 1e-8
    )

    x_norm = torch.linalg.vector_norm(x, dim=-1)
    y_norm = torch.linalg.vector_norm(y, dim=-1)

    return (
        cosine_similarity,
        euclidean_distance,
        normalized_euclidean,
        x_norm,
        y_norm,
    )


# ============================================================
# Load activations
# ============================================================

x = torch.load(DATA_DIR / "difference_in_means" / "pos_act.pt").float()

y = torch.load(DATA_DIR / "difference_in_means" / "neg_act.pt").float()

if x.ndim == 2:
    x = x.unsqueeze(0)  # [1, layers, hidden]
if y.ndim == 2:
    y = y.unsqueeze(0)  # [1, layers, hidden]

print("=" * 80)
print("ACTIVATION SHAPES")
print("=" * 80)
print(f"x: {x.shape}")
print(f"y: {y.shape}")

assert x.shape == y.shape

num_tokens, num_layers, hidden_dim = x.shape

print(f"\nTokens:     {num_tokens}")
print(f"Layers:     {num_layers}")
print(f"Hidden dim: {hidden_dim}")


# ============================================================
# Compute metrics
# ============================================================

(
    cosine_similarity,
    euclidean_distance,
    normalized_euclidean,
    x_norm,
    y_norm,
) = compare_act(x, y)

print("\nMetric shapes:")
print(f"cosine similarity:    {cosine_similarity.shape}")
print(f"euclidean distance:   {euclidean_distance.shape}")
print(f"normalized euclidean: {normalized_euclidean.shape}")


# ============================================================
# Global summary
# ============================================================


def print_summary(name, values):
    print(f"\n{name}")
    print("-" * 80)
    print(f"mean:   {values.mean().item():.6f}")
    print(f"std:    {values.std(correction=0).item():.6f}")
    print(f"median: {values.median().item():.6f}")
    print(f"min:    {values.min().item():.6f}")
    print(f"max:    {values.max().item():.6f}")


print("\n")
print("=" * 80)
print("GLOBAL METRICS")
print("=" * 80)

print_summary("COSINE SIMILARITY", cosine_similarity)
print_summary("EUCLIDEAN DISTANCE", euclidean_distance)
print_summary("NORMALIZED EUCLIDEAN DISTANCE", normalized_euclidean)
print_summary("POS ACTIVATION NORM", x_norm)
print_summary("NEG ACTIVATION NORM", y_norm)


# ============================================================
# Per-layer statistics
# ============================================================

print("\n")
print("=" * 130)
print("PER-LAYER METRICS")
print("=" * 130)

header = (
    f"{'Layer':>5} | "
    f"{'Cos mean':>10} {'Cos std':>10} | "
    f"{'Euc mean':>10} {'Euc std':>10} | "
    f"{'NormEuc':>10} | "
    f"{'Pos norm':>10} {'Neg norm':>10}"
)

print(header)
print("-" * 130)

layer_cos_mean = []
layer_cos_std = []

layer_euc_mean = []
layer_euc_std = []

layer_neuc_mean = []
layer_neuc_std = []

layer_xnorm_mean = []
layer_ynorm_mean = []


for layer in range(num_layers):
    cos = cosine_similarity[:, layer]
    euc = euclidean_distance[:, layer]
    neuc = normalized_euclidean[:, layer]
    xn = x_norm[:, layer]
    yn = y_norm[:, layer]

    cos_mean = cos.mean().item()
    cos_std = cos.std(correction=0).item()

    euc_mean = euc.mean().item()
    euc_std = euc.std(correction=0).item()

    neuc_mean = neuc.mean().item()
    neuc_std = neuc.std(correction=0).item()

    xn_mean = xn.mean().item()
    yn_mean = yn.mean().item()

    print(
        f"{layer:5d} | "
        f"{cos_mean:10.6f} {cos_std:10.6f} | "
        f"{euc_mean:10.6f} {euc_std:10.6f} | "
        f"{neuc_mean:10.6f} | "
        f"{xn_mean:10.6f} {yn_mean:10.6f}"
    )

    layer_cos_mean.append(cos_mean)
    layer_cos_std.append(cos_std)

    layer_euc_mean.append(euc_mean)
    layer_euc_std.append(euc_std)

    layer_neuc_mean.append(neuc_mean)
    layer_neuc_std.append(neuc_std)

    layer_xnorm_mean.append(xn_mean)
    layer_ynorm_mean.append(yn_mean)


# ============================================================
# Most / least similar layers
# ============================================================

cos_mean = torch.tensor(layer_cos_mean)
euc_mean = torch.tensor(layer_euc_mean)
neuc_mean = torch.tensor(layer_neuc_mean)

most_similar_layer = torch.argmax(cos_mean).item()
least_similar_layer = torch.argmin(cos_mean).item()

most_different_euc_layer = torch.argmax(euc_mean).item()
least_different_euc_layer = torch.argmin(euc_mean).item()

most_different_norm_layer = torch.argmax(neuc_mean).item()

print("\n")
print("=" * 80)
print("LAYER INTERPRETATION")
print("=" * 80)

print(
    f"Highest cosine similarity: "
    f"layer {most_similar_layer} "
    f"({cos_mean[most_similar_layer]:.6f})"
)

print(
    f"Lowest cosine similarity:  "
    f"layer {least_similar_layer} "
    f"({cos_mean[least_similar_layer]:.6f})"
)

print(
    f"Highest Euclidean distance: "
    f"layer {most_different_euc_layer} "
    f"({euc_mean[most_different_euc_layer]:.6f})"
)

print(
    f"Lowest Euclidean distance:  "
    f"layer {least_different_euc_layer} "
    f"({euc_mean[least_different_euc_layer]:.6f})"
)

print(
    f"Highest normalized Euclidean: "
    f"layer {most_different_norm_layer} "
    f"({neuc_mean[most_different_norm_layer]:.6f})"
)


# ============================================================
# Plot 1: Mean cosine similarity across layers
# ============================================================

layers = torch.arange(num_layers)

plt.figure(figsize=(12, 6))

plt.plot(layers, layer_cos_mean, marker="o")

plt.xlabel("Layer")
plt.ylabel("Cosine similarity")
plt.title("Mean Cosine Similarity Across Tokens")

plt.axhline(1.0, linestyle="--", linewidth=1)

plt.xticks(layers)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: Mean Euclidean distance across layers
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(layers, layer_euc_mean, marker="o")

plt.xlabel("Layer")
plt.ylabel("Euclidean distance")
plt.title("Mean Euclidean Distance Across Tokens")

plt.xticks(layers)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 3: Normalized Euclidean distance
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(layers, layer_neuc_mean, marker="o")

plt.xlabel("Layer")
plt.ylabel("Normalized Euclidean distance")
plt.title("Mean Normalized Euclidean Distance Across Tokens")

plt.xticks(layers)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 4: Activation norms
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(layers, layer_xnorm_mean, marker="o", label="Positive")
plt.plot(layers, layer_ynorm_mean, marker="o", label="Negative")

plt.xlabel("Layer")
plt.ylabel("Activation norm")
plt.title("Mean Activation Norm Across Tokens")

plt.xticks(layers)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================
# Plot 5: Cosine similarity heatmap
# ============================================================

plt.figure(figsize=(14, 10))

plt.imshow(
    cosine_similarity.detach().cpu().numpy(),
    aspect="auto",
    interpolation="nearest",
)

plt.colorbar(label="Cosine similarity")

plt.xlabel("Layer")
plt.ylabel("Token")
plt.title("Token × Layer Cosine Similarity")

plt.xticks(range(num_layers))

plt.tight_layout()
plt.show()


# ============================================================
# Plot 6: Euclidean distance heatmap
# ============================================================

plt.figure(figsize=(14, 10))

plt.imshow(
    euclidean_distance.detach().cpu().numpy(),
    aspect="auto",
    interpolation="nearest",
)

plt.colorbar(label="Euclidean distance")

plt.xlabel("Layer")
plt.ylabel("Token")
plt.title("Token × Layer Euclidean Distance")

plt.xticks(range(num_layers))

plt.tight_layout()
plt.show()


# ============================================================
# Plot 7: Normalized Euclidean heatmap
# ============================================================

plt.figure(figsize=(14, 10))

plt.imshow(
    normalized_euclidean.detach().cpu().numpy(),
    aspect="auto",
    interpolation="nearest",
)

plt.colorbar(label="Normalized Euclidean distance")

plt.xlabel("Layer")
plt.ylabel("Token")
plt.title("Token × Layer Normalized Euclidean Distance")

plt.xticks(range(num_layers))

plt.tight_layout()
plt.show()


# ============================================================
# Token-level analysis
# ============================================================

token_cos_mean = cosine_similarity.mean(dim=1)
token_euc_mean = euclidean_distance.mean(dim=1)
token_neuc_mean = normalized_euclidean.mean(dim=1)

print("\n")
print("=" * 80)
print("TOKEN-LEVEL SUMMARY")
print("=" * 80)

print(
    f"Most divergent token by cosine: "
    f"{torch.argmin(token_cos_mean).item()} "
    f"(similarity={token_cos_mean.min().item():.6f})"
)

print(
    f"Most divergent token by Euclidean: "
    f"{torch.argmax(token_euc_mean).item()} "
    f"(distance={token_euc_mean.max().item():.6f})"
)

print(
    f"Most divergent token by normalized Euclidean: "
    f"{torch.argmax(token_neuc_mean).item()} "
    f"(distance={token_neuc_mean.max().item():.6f})"
)

print("\n")
print("=" * 100)
print("MOST CHANGED TOKEN AT EACH LAYER")
print("=" * 100)

for layer in range(num_layers):
    cos_values = cosine_similarity[:, layer]
    euc_values = euclidean_distance[:, layer]
    neuc_values = normalized_euclidean[:, layer]

    cos_token = torch.argmin(cos_values)
    euc_token = torch.argmax(euc_values)
    neuc_token = torch.argmax(neuc_values)

    print(
        f"Layer {layer:2d} | "
        f"cosine: token {cos_token.item():3d} "
        f"({cos_values[cos_token].item():.4f}) | "
        f"euclidean: token {euc_token.item():3d} "
        f"({euc_values[euc_token].item():.2f}) | "
        f"norm-euc: token {neuc_token.item():3d} "
        f"({neuc_values[neuc_token].item():.4f})"
    )

token = 0

plt.figure(figsize=(10, 5))

plt.plot(
    layers,
    cosine_similarity[token].detach().cpu(),
    marker="o",
)

plt.xlabel("Layer")
plt.ylabel("Cosine similarity")
plt.title(f"Token {token}: Cosine Similarity Across Layers")
plt.xticks(layers)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))

plt.plot(
    layers,
    normalized_euclidean[token].detach().cpu(),
    marker="o",
)

plt.xlabel("Layer")
plt.ylabel("Normalized Euclidean distance")
plt.title(f"Token {token}: Normalized Distance Across Layers")
plt.xticks(layers)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
