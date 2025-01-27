import matplotlib.pyplot as plt

marker = "."
markersize = 8
y_label = "tokens/sec"
x_label = "batch size"
title = "batch size = {batch_size}, 70B model"
tp_world_size = [2, 4, 8]


def plot(
    batch_size: int,
    standard: list[float],
    ladder: list[float],
    upper_bound: list[float],
    parallel: list[float],
    extra_title: str,
) -> None:
    plt.figure()

    plt.plot(tp_world_size, standard, marker=marker, markersize=markersize, label="standard transformer")
    plt.plot(tp_world_size, ladder, marker=marker, markersize=markersize, label="ladder transformer")
    plt.plot(tp_world_size, parallel, marker=marker, markersize=markersize, label="parallel attn")
    plt.plot(tp_world_size, upper_bound, marker=marker, markersize=markersize, label="upper bound")

    plt.xticks(tp_world_size)
    plt.xlim(1.5, 8.5)

    plt.title(title.format(batch_size=batch_size) + f", {extra_title}")
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.grid(True, linestyle=":", color="gray", linewidth=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()

    plt.savefig(f"70b-{batch_size}-{'-'.join(extra_title.split())}.png", dpi=300)

extra_title = "NVLink enabled"

plot(
    batch_size=1,
    standard=[35.42, 59.41, 77.39],
    ladder=[36.69, 67.51, 101.22],
    upper_bound=[38.24, 69.04, 110.59],
    parallel=[36.67, 65.4, 94.22],
    extra_title=extra_title,
)

plot(
    batch_size=4,
    standard=[120.58, 185.01, 258.56],
    ladder=[126.09, 204.3, 331.45],
    upper_bound=[130.77, 213.73, 355.8],
    parallel=[123.66, 201.62, 307.34],
    extra_title=extra_title,
)

plot(
    batch_size=16,
    standard=[float("nan"), 585.23, 843.15],
    ladder=[float("nan"), 635.98, 1003.52],
    upper_bound=[float("nan"), 665.07, 1109.65],
    parallel=[float("nan"), 628.23, 973.74],
    extra_title=extra_title,
)

plot(
    batch_size=64,
    standard=[float("nan"), 1249.67, 1940.99],
    ladder=[float("nan"), 1358.65, 2242.1],
    upper_bound=[float("nan"), 1433.53, 2474.49],
    parallel=[float("nan"), 1311.54, 2259.38],
    extra_title=extra_title,
)

extra_title = "NVLink disabled"

plot(
    batch_size=1,
    standard=[33.77, 53.35, 51.66],
    ladder=[36.94, 66.6, 82.59],
    upper_bound=[37.87, 69.25, 108.86],
    parallel=[36.12, 61.93, 72.36],
    extra_title=extra_title,
)

plot(
    batch_size=4,
    standard=[106.6, 158.77, 173.62],
    ladder=[125.55, 204.24, 271.82],
    upper_bound=[130.69, 223.64, 360.43],
    parallel=[116.83, 183.53, 241.08],
    extra_title=extra_title,
)

plot(
    batch_size=16,
    standard=[float("nan"), 518.41, 546.68],
    ladder=[float("nan"), 598.71, 738.56],
    upper_bound=[float("nan"), 683.66, 1122.79],
    parallel=[float("nan"), 583.48, 744.33],
    extra_title=extra_title,
)

plot(
    batch_size=64,
    standard=[float("nan"), 1199.62, 1454.42],
    ladder=[float("nan"), 1313.44, 1864.05],
    upper_bound=[float("nan"), 1450.27, 2489.51],
    parallel=[float("nan"), 1276.66, 1873.71],
    extra_title=extra_title,
)
