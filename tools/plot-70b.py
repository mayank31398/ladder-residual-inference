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
    plt.savefig(f"70b-{batch_size}.png", dpi=300)

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
    standard=[float("nan"), 335.81, 475.04],
    ladder=[float("nan"), 373.33, 596.73],
    upper_bound=[float("nan"), 389.03, 646.18],
    parallel=[float("nan"), 363.29, 555.88],
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
