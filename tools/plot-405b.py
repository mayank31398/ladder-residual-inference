import matplotlib.pyplot as plt

marker = "."
markersize = 8
y_label = "tokens/sec speedup"
x_label = "batch size"


batch_sizes = [1, 4, 8, 16]
ladder_nvl = [1.364019677, 1.307557841, 1.393313367, 1.349331588]
parallel_nvl = [1.237526353, 1.242467866, 1.285679359, 1.272326351]
upper_bound_nvl = [1.52073085, 1.511876607, 1.658883931, 1.631580761]

ladder_no_nvl = [1.489855072, 1.565769112, 1.434301985, 1.461046963]
parallel_no_nvl = [1.396618357, 1.439028165, 1.415040616, 1.398749487]
upper_bound_no_nvl = [2.106763285, 2.274682761, 2.270580353, 2.15964584]

plt.plot(batch_sizes, ladder_nvl, marker=marker, markersize=markersize)
plt.plot(batch_sizes, parallel_nvl, marker=marker, markersize=markersize)
plt.plot(batch_sizes, upper_bound_nvl, marker=marker, markersize=markersize)

plt.ylim(1, 2.5)
plt.xticks(batch_sizes)

plt.title("NVLink enabled (TP = 16)")
plt.ylabel(y_label)
plt.xlabel(x_label)
plt.grid(True, linestyle=":", color="gray", linewidth=0.5)
plt.savefig("405b-nvl.png", dpi=300)


plt.figure()

plt.plot(batch_sizes, ladder_no_nvl, marker=marker, markersize=markersize)
plt.plot(batch_sizes, parallel_no_nvl, marker=marker, markersize=markersize)
plt.plot(batch_sizes, upper_bound_no_nvl, marker=marker, markersize=markersize)

plt.ylim(1, 2.5)
plt.xticks(batch_sizes)

plt.title("NVLink disabled (TP = 16)")
plt.ylabel(y_label)
plt.xlabel(x_label)
plt.grid(True, linestyle=":", color="gray", linewidth=0.5)
plt.savefig("405b-no-nvl.png", dpi=300)
