import matplotlib.pyplot as plt

marker = "."
markersize = 8
y_label = "tokens/sec"
x_label = "batch size"
title = "batch size = {batch_size}, 70B model"
tp_world_size = [2, 4, 8]

blue = "#1f77b4"
orange = "#ff7f0e"
green = "#2ca02c"
red = "#d62728"


def plot(
    batch_size: int,
    standard_nvl: list[float],
    ladder_nvl: list[float],
    upper_bound_nvl: list[float],
    parallel_nvl: list[float],
    standard_no_nvl: list[float],
    ladder_no_nvl: list[float],
    parallel_no_nvl: list[float],
) -> None:
    plt.figure()

    plt.plot(tp_world_size, standard_nvl, marker=marker, markersize=markersize, label="standard transformer P2P=1", linestyle="-", color=blue)
    plt.plot(tp_world_size, standard_no_nvl, marker=marker, markersize=markersize, label="standard transformer P2P=0", linestyle="--", color=blue)
    plt.plot(tp_world_size, ladder_nvl, marker=marker, markersize=markersize, label="ladder transformer P2P=1", linestyle="-", color=orange)
    plt.plot(tp_world_size, ladder_no_nvl, marker=marker, markersize=markersize, label="ladder transformer P2P=0", linestyle="--", color=orange)
    plt.plot(tp_world_size, parallel_nvl, marker=marker, markersize=markersize, label="parallel attn P2P=1", linestyle="-", color=green)
    plt.plot(tp_world_size, parallel_no_nvl, marker=marker, markersize=markersize, label="parallel attn P2P=0", linestyle="--", color=green)
    plt.plot(tp_world_size, upper_bound_nvl, marker=marker, markersize=markersize, label="upper bound P2P=1", linestyle="-", color=red)

    plt.xticks(tp_world_size)
    plt.xlim(1.5, 8.5)

    plt.title(title.format(batch_size=batch_size))
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid(True, linestyle=":", color="gray", linewidth=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(f"70b-{batch_size}.png", dpi=300)

plot(
    batch_size=1,
    standard_nvl=[35.42, 59.41, 77.39],
    ladder_nvl=[36.69, 67.51, 101.22],
    upper_bound_nvl=[38.24, 69.04, 110.59],
    parallel_nvl=[36.67, 65.4, 94.22],
    standard_no_nvl=[33.77, 53.35, 51.66],
    ladder_no_nvl=[36.94, 66.6, 82.59],
    parallel_no_nvl=[36.12, 61.93, 72.36],
)

plot(
    batch_size=4,
    standard_nvl=[120.58, 185.01, 258.56],
    ladder_nvl=[126.09, 204.3, 331.45],
    upper_bound_nvl=[130.77, 213.73, 355.8],
    parallel_nvl=[123.66, 201.62, 307.34],
    standard_no_nvl=[106.6, 158.77, 173.62],
    ladder_no_nvl=[125.55, 204.24, 271.82],
    parallel_no_nvl=[116.83, 183.53, 241.08],
)

plot(
    batch_size=16,
    standard_nvl=[float("nan"), 585.23, 843.15],
    ladder_nvl=[float("nan"), 635.98, 1003.52],
    upper_bound_nvl=[float("nan"), 665.07, 1109.65],
    parallel_nvl=[float("nan"), 628.23, 973.74],
    standard_no_nvl=[float("nan"), 518.41, 546.68],
    ladder_no_nvl=[float("nan"), 598.71, 738.56],
    parallel_no_nvl=[float("nan"), 583.48, 744.33],
)

plot(
    batch_size=64,
    standard_nvl=[float("nan"), 1249.67, 1940.99],
    ladder_nvl=[float("nan"), 1358.65, 2242.1],
    upper_bound_nvl=[float("nan"), 1433.53, 2474.49],
    parallel_nvl=[float("nan"), 1311.54, 2259.38],
    standard_no_nvl=[float("nan"), 1199.62, 1454.42],
    ladder_no_nvl=[float("nan"), 1313.44, 1864.05],
    parallel_no_nvl=[float("nan"), 1276.66, 1873.71],
)
