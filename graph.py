import json
import matplotlib.pyplot as plt
import numpy as np
import os
from path import path_join
from constants import json_output_folder, graph_output_folder, style_list, loss_list

method_list = ["AdaIN", "RAASN"]

bar_width = 0.2

os.makedirs(graph_output_folder, exist_ok=True)

# graph for all methods on each style
for style in style_list:
    for loss_type in loss_list:

        # get data from json
        mean_list = []
        standard_deviation_list = []
        for method in method_list:
            json_file_path = path_join(json_output_folder, method)
            with open(f"{json_file_path}/{style}-{loss_type}.json", 'r') as json_file:
                data = json.load(json_file)
            mean = np.mean(data)
            standard_deviation = np.std(data)
            mean_list.append(mean)
            standard_deviation_list.append(standard_deviation)

        # generate graph
        bar_positions = np.arange(len(method_list))
        plt.bar(x=bar_positions, height=mean_list, width=bar_width,
                yerr=standard_deviation_list, align="center", alpha=1, ecolor='black', capsize=10)
        plt.xticks(ticks=bar_positions, labels=method_list, fontsize=15)

        # Set plot title and labels for x-axis & y-axis
        plt.title(label=style, fontsize=20)
        plt.xlabel("method")
        plt.ylabel(f"{loss_type}")

        # save image
        graph_name = f"{style}_{loss_type}.svg"
        plt.savefig(f"{graph_output_folder}/{graph_name}", format="svg")
        print(f"generated: {graph_name}")

        # clear plot
        plt.clf()

# summary graph
