# %%
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


import os
import shutil

from os.path import join

from os.path import join, basename, split

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import numpy as np


# %%

def main(simulation_name, output_folder):

    input_folder = f"{simulation_name}.out"

    df = pd.read_csv(join(input_folder, 'table.txt'),sep='\t')
    title = basename(simulation_name)

    t = df["# t (s)"].to_numpy() * 1e9
    pos_x = df["ext_bubbleposx (m)"] * 1e9
    pos_y = df["ext_bubbleposy (m)"] * 1e9




    plt.plot(t, pos_x, label="x")
    plt.xlabel("time (ns)")
    plt.ylabel("position (nm)")
    plt.title("Skyrmion Position in X-Direction")
    plt.savefig(join(output_folder, f"{title}_skyrmion_position.png"), dpi=300)
    plt.close()

    plt.plot(t, pos_y, label="y")
    plt.xlabel("time (ns)")
    plt.ylabel("position (nm)")
    plt.title("Skyrmion position in Y-Direction")
    plt.savefig(join(output_folder, f"{title}_skyrmion_position_y.png"), dpi=300)
    plt.close()


    speed_x = np.gradient(pos_x, t)
    speed_y = np.gradient(pos_y, t)

    plt.plot(t, speed_x, label="speed x")
    plt.xlabel("time (ns)")
    plt.ylabel("speed (nm/ns)")
    plt.title("Skyrmion Speed in X-Direction")
    plt.savefig(join(output_folder, f"{title}_skyrmion_speed.png"), dpi=300)
    plt.close()


    plt.plot(t, speed_y, label="speed y")
    plt.xlabel("time (ns)")
    plt.ylabel("speed (nm/ns)")
    plt.title("Skyrmion speed in Y-Direction")
    plt.savefig(join(output_folder, f"{title}_skyrmion_speed_y.png"), dpi=300)
    plt.close()

    df_out = pd.DataFrame(
        {
            "time": t,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "speed_x": speed_x,
            "speed_y": speed_y,
        }
    )

    df_out.to_excel(join(output_folder, f"{title}_skyrmion_data.xlsx"), index=False)
    return 




# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot skyrmion data")
    parser.add_argument(
        "simulation_name", type=str, help="name of the simulation folder"
    )
    parser.add_argument(
        "output_folder", type=str, help="name of the output folder"
    )

    args = parser.parse_args()

    simulation_name = args.simulation_name
    # output_folder = simulation_name + ".out/" + args.output_folder

    output_folder = join(split(args.simulation_name)[0], args.output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main(simulation_name, output_folder)


