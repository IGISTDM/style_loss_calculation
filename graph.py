import json
import matplotlib.pyplot as plt
import numpy as np
import os
from path import path_join
from constants import output_folder, graph_output_folder, style_list

method_list = ["RAASN"]

#types = ["content", "style"]
loss_list = ["content"]

os.makedirs(graph_output_folder, exist_ok=True)

for method in method_list:
    for loss_type in loss_list:
        mean_list = []
        standard_deviation_list = []
        for style in style_list:
            json_file_path = path_join(output_folder, method)
            with open(f"{json_file_path}/{style}-{loss_type}.json", 'r') as json_file:
                data = json.load(json_file)

            mean = np.mean(data)
            standard_deviation = np.std(data)
            mean_list.append(mean)
            standard_deviation_list.append(standard_deviation)

        # x coordinates of the bars
        bar_positions = np.arange(len(style_list))

        # Build the plot
        fig, ax = plt.subplots()
        ax.bar(bar_positions, mean_list, yerr=standard_deviation_list, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel(f"{loss_type}")
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(style_list)
        ax.set_title(f"{method}")
        ax.yaxis.grid(True)

        # save image
        graph_name = f"{method}-{loss_type}.png"
        plt.savefig(f"{graph_output_folder}/{graph_name}")
        print(f"generated: {graph_name}")