import os
import shutil

from os.path import join

import pandas as pd
import numpy as np

from scipy import interpolate
from scipy.fft import fftfreq, fft

from os.path import join, basename, split

import matplotlib.pyplot as plt
import argparse



def main(simulation_name, output_folder,
        target_freq_lower, target_freq_upper,  
         no_regions=1, fft_ylim2=5e4):
    
    input_folder = f"{simulation_name}.out"
    # output_folder = f""
    title = basename(simulation_name)
    
    df = pd.read_csv(join(input_folder, 'table.txt'),sep='\t')
    t = df['# t (s)'].to_numpy()
    
    magnetizations = []
    for i in range(1, no_regions+1):
        magnetizations.append(df['m.region{}y ()'.format(i)].to_numpy())
        
    
    # plot the data 
    plot_mag(magnetizations, t, output_folder, no_regions, 
             f'Reduced Magnetization of {title}')
    
    

    # FFT
    peaks = fft_function(magnetizations, t, no_regions,
                         target_freq_lower, target_freq_upper,
                         output_folder, title, fft_ylim=fft_ylim2)


    
    return peaks
        
def plot_mag(magnetizations, t, output_folder, no_regions, title):

    
        
    for i in range(no_regions):
        plt.plot(t, magnetizations[i], label=f'Region {i+1}')


    plt.legend()
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Reduced Magnetization')
    plt.grid()
    plt.savefig(join(output_folder, f'{title}.png'), dpi=300)
    plt.close()
    
    return 



    
def fft_function(magnetizations, t, no_regions,
                 target_freq_lower, target_freq_upper,
                 output_folder, title, fft_ylim=None):

    t_new = np.arange(t[0], t[-1], t[1] - t[0])

    t_new_len = len(t_new)
    xf = fftfreq(t_new_len, t_new[1]-t_new[0])


    magnetizations_interp = np.zeros((no_regions, t_new_len))
    yf = np.zeros((no_regions, t_new_len), dtype=np.complex_)
    peaks = np.zeros(no_regions)

    for i in range(no_regions):
        f = interpolate.interp1d(t, magnetizations[i])
        magnetizations_interp[i] = f(t_new)
        yf[i] = fft(magnetizations_interp[i])

        plt.plot(xf, np.abs(yf[i]), label=f'Region {i+1}')
        peaks[i] = xf[np.argmax(np.abs(yf[i][10:int(len(yf[i])/2)]))+10]/1e9

    

    if fft_ylim is None:
        fft_ylim = np.max(np.abs(yf)) * 1.1



    plt.xlim(target_freq_lower, target_freq_upper)
    plt.ylim(0, fft_ylim)
    plt.title(f'FFT of {title}')
    plt.ylabel('Amplitude (a.u.)')
    plt.xlabel('Frequency (0.1GHz)')
    plt.legend()
    plt.savefig(join(output_folder, f'FFT_freq={title}.png'), dpi=300)
    plt.close()
    
    return peaks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation_name', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--target_freq_lower', type=float)
    parser.add_argument('--target_freq_upper', type=float)
    parser.add_argument('--no_regions', type=int)
    parser.add_argument('--fft_ylim2', type=float)

    args = parser.parse_args()
    


    args.output_folder = join(split(args.simulation_name)[0], args.output_folder)

    try:
        os.mkdir(args.output_folder)
    except:
        pass


    peaks = main(args.simulation_name, args.output_folder,
                args.target_freq_lower, args.target_freq_upper,
                args.no_regions, args.fft_ylim2)
    
    with open(join(args.output_folder, f'{basename(args.simulation_name)}.txt'), 'w') as f:
        for item in peaks:
            f.write(f"{item}\n")