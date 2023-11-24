from glob import glob
from os.path import join, basename, splitext

import pickle
from matplotlib import pyplot as plt
import numpy as np

from os import mkdir

from misc import *
import argparse


def find_nearest(array, value):
    return (np.abs(array - value)).argmin()


def main_m(no_files, npy_folder, output_dir, target_freq, preflix, cut_off,):

    def plot_spectrum(xf, fft_spectrum, peaks):
        plt.plot(xf, fft_spectrum)
        plt.plot(xf[peaks], fft_spectrum[peaks], "x")
        
        # plt.xlim(0, xf.max())
        plt.xlim(0, cut_off)
        plt.ylim(0, fft_spectrum[1:len(fft_spectrum)//2].max() * 1.1)

        plt.savefig(join(output_dir, preflix + '_fft_spectrum' + '.png'),
                    dpi=300)
        plt.close()
        
        np.save(join(output_dir, preflix + '_fft_spectrum_xf.npy'), xf)
        np.save(join(output_dir, preflix + '_fft_spectrum_fft_spectrum.npy'), fft_spectrum)
        
        
    x = np.load(join(npy_folder, 'x.npy'))
    y = np.load(join(npy_folder, 'y.npy'))
    z = np.load(join(npy_folder, 'z.npy'))
    
    t = np.load(join(npy_folder, 't.npy'))
    shape = np.load(join(npy_folder, 'shape.npy'))
    vaccum = np.load(join(npy_folder, 'vaccum.npy'))
    

    x = x[:,:,:,0]
    y = y[:,:,:,0]
    z = z[:,:,:,0]
    vaccum = vaccum[:,:,0]


    xf, fft_spectrum, peaks, x_fft, \
    y_fft, z_fft = fft_function_m(x, y, z, t, shape, no_files, target_freq)
     
    plot_spectrum(xf, fft_spectrum, peaks)

    np.save(join(npy_folder, preflix + '_x_fft.npy'), x_fft)
    np.save(join(npy_folder, preflix + '_y_fft.npy'), y_fft)
    np.save(join(npy_folder, preflix + '_z_fft.npy'), z_fft)

    power = y_fft.real**2 + x_fft.real**2 + z_fft.real**2
       


    peak_freq = []
    
    for peak in peaks:
        freq_ = xf[peak]
        
        # if freq_ > 0:
        if 0 < freq_ < cut_off:
            peak_freq.append(freq_)
            
            to_plot = y_fft[:,peak].reshape(shape[0], shape[1]).real**2 + \
                    x_fft[:,peak].reshape(shape[0], shape[1]).real**2 + \
                    z_fft[:,peak].reshape(shape[0], shape[1]).real**2
                    
            to_plot[vaccum] = np.nan
            
            # to_plot = np.log(to_plot)
            
            phase = np.angle(y_fft[:,peak] + x_fft[:,peak] + z_fft[:,peak])
            phase = phase.reshape(shape[0], shape[1])
            phase[vaccum] = np.nan
            

            
            freq_title = str(round(xf[peak]/1e9, 4)) + ' GHz'

            np.save(join(output_dir, preflix + '_single_peak_phase' + freq_title + '.npy'), phase)
            np.save(join(output_dir, preflix + '_single_peak_power' + freq_title + '.npy'), to_plot)
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im = ax[0].imshow(to_plot, cmap='jet')
            
            ax[0].set_title('Power ' + freq_title)
            plt.colorbar(im, ax=ax[0])

            im = ax[1].imshow(phase, vmin=-np.pi, vmax=np.pi, cmap=plt.cm.hsv)
            ax[1].set_title('Phase ' + freq_title)
            plt.colorbar(im, ax=ax[1])
            plt.savefig(join(output_dir, preflix + '_single_peak_' + freq_title + '.png'), dpi=300)
            plt.close()
                    
    for peak in peaks:
        
        left = int(peak*0.95)
        right = int(peak*1.05)
        
        freq_ = xf[peak]
        
        # if freq_ > 0:
        
        if 0 < freq_ < cut_off:
            peak_freq.append(freq_)
            
            to_plot = np.zeros((shape[0], shape[1]))
            phase = np.zeros(int(shape[0]*shape[1]))
            
            # print(left, peak, right)
            
            for i in range(left, right):
                to_plot += y_fft[:,i].reshape(shape[0], shape[1]).real**2 + \
                    x_fft[:,i].reshape(shape[0], shape[1]).real**2 + \
                    z_fft[:,i].reshape(shape[0], shape[1]).real**2
                    
                phase += np.angle(y_fft[:,i] + x_fft[:,i] + z_fft[:,i])
                    
                
                    
            to_plot[vaccum] = np.nan
            phase = phase.reshape(shape[0], shape[1])
            phase[vaccum] = np.nan
            
            # to_plot = np.log(to_plot)
            
            freq_title = str(round(xf[left]/1e9, 4)) + '-' + \
                str(round(xf[right]/1e9, 4)) + ' GHz'

            np.save(join(output_dir, preflix + '_range_power' + freq_title + '.npy'), to_plot)
            np.save(join(output_dir, preflix + '_range_phase' + freq_title + '.npy'), phase)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im = ax[0].imshow(to_plot, cmap='jet')
            
            ax[0].set_title('Power ' + freq_title)
            plt.colorbar(im, ax=ax[0])

            im = ax[1].imshow(phase, vmin=-np.pi, vmax=np.pi, cmap=plt.cm.hsv)
            ax[1].set_title('Phase ' + freq_title)
            plt.colorbar(im, ax=ax[1])
            plt.savefig(join(output_dir, preflix + '_range_' + freq_title + '.png'), dpi=300)
            plt.close()
    
    with open(join(output_dir, preflix + '_peaks.txt'), 'w') as f:
        for item in peak_freq:
            f.write("%s\n" % item)
            
    return 



def main_demag(no_files, npy_folder, m_npy_folder, output_dir, target_freq, preflix, cut_off):

    def plot_spectrum(xf, fft_spectrum, peaks):
        plt.plot(xf, fft_spectrum)
        plt.plot(xf[peaks], fft_spectrum[peaks], "x")
        # plt.xlim(0, xf.max())
        plt.xlim(0, cut_off)
        # plt.ylim(0, fft_spectrum.max())
        plt.ylim(0, fft_spectrum[1:len(fft_spectrum)//2].max())

        plt.savefig(join(output_dir, preflix + '_fft_spectrum' + '.png'),
                    dpi=300)
        plt.close()
        
        np.save(join(output_dir, preflix + '_fft_spectrum_xf.npy'), xf)
        np.save(join(output_dir, preflix + '_fft_spectrum_fft.npy'), fft_spectrum)
        
        
    x = np.load(join(npy_folder, 'x.npy'))
    y = np.load(join(npy_folder, 'y.npy'))
    z = np.load(join(npy_folder, 'z.npy'))
    
    t = np.load(join(npy_folder, 't.npy'))
    shape = np.load(join(npy_folder, 'shape.npy'))
    vaccum = np.load(join(m_npy_folder, 'vaccum.npy'))

    vaccum = np.bitwise_not(vaccum)


    x = x[:,:,:,0]
    y = y[:,:,:,0]
    z = z[:,:,:,0]
    vaccum = vaccum[:,:,0]

    
    xf, fft_spectrum, peaks, x_fft, \
    y_fft, z_fft = fft_function_m(x, y, z, t, shape, no_files, target_freq)
     
    plot_spectrum(xf, fft_spectrum, peaks)
    
    np.save(join(npy_folder, preflix + '_x_fft.npy'), x_fft)
    np.save(join(npy_folder, preflix + '_y_fft.npy'), y_fft)
    np.save(join(npy_folder, preflix + '_z_fft.npy'), z_fft)

    power = y_fft.real**2 + x_fft.real**2 + z_fft.real**2
        
    peak_freq = []
    
    for peak in peaks:
        freq_ = xf[peak]
        
        # if freq_ > 0:
        if 0 < freq_ < cut_off:
            peak_freq.append(freq_)
            
            to_plot = y_fft[:,peak].reshape(shape[0], shape[1]).real**2 + \
                    x_fft[:,peak].reshape(shape[0], shape[1]).real**2 + \
                    z_fft[:,peak].reshape(shape[0], shape[1]).real**2
                    
            to_plot[vaccum] = np.nan
            
            # to_plot = np.log(to_plot)
            
            phase = np.angle(y_fft[:,peak] + x_fft[:,peak] + z_fft[:,peak])
            phase = phase.reshape(shape[0], shape[1])
            phase[vaccum] = np.nan
            

            
            freq_title = str(round(xf[peak]/1e9, 4)) + ' GHz'

            np.save(join(output_dir, preflix + '_single_peak_phase' + freq_title + '.npy'), phase)
            np.save(join(output_dir, preflix + '_single_peak_power' + freq_title + '.npy'), to_plot)
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im = ax[0].imshow(to_plot, cmap='jet')
            
            ax[0].set_title('Power ' + freq_title)
            plt.colorbar(im, ax=ax[0])

            im = ax[1].imshow(phase, vmin=-np.pi, vmax=np.pi, cmap=plt.cm.hsv)
            ax[1].set_title('Phase ' + freq_title)
            plt.colorbar(im, ax=ax[1])
            plt.savefig(join(output_dir, preflix + '_single_peak_' + freq_title + '.png'), dpi=300)
            plt.close()
                    
    for peak in peaks:
        
        left = int(peak*0.95)
        right = int(peak*1.05)
        
        freq_ = xf[peak]
        
        # if freq_ > 0:
        if 0 < freq_ < cut_off:
            peak_freq.append(freq_)
            
            to_plot = np.zeros((shape[0], shape[1]))
            phase = np.zeros(int(shape[0]*shape[1]))
            
            # print(left, peak, right)
            
            for i in range(left, right):
                to_plot += y_fft[:,i].reshape(shape[0], shape[1]).real**2 + \
                    x_fft[:,i].reshape(shape[0], shape[1]).real**2 + \
                    z_fft[:,i].reshape(shape[0], shape[1]).real**2
                    
                phase += np.angle(y_fft[:,i] + x_fft[:,i] + z_fft[:,i])
                    
                
                    
            to_plot[vaccum] = np.nan
            phase = phase.reshape(shape[0], shape[1])
            phase[vaccum] = np.nan
            
            # to_plot = np.log(to_plot)
            
            freq_title = str(round(xf[left]/1e9, 4)) + '-' + \
                str(round(xf[right]/1e9, 4)) + ' GHz'

            np.save(join(output_dir, preflix + '_range_power' + freq_title + '.npy'), to_plot)
            np.save(join(output_dir, preflix + '_range_phase' + freq_title + '.npy'), phase)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            im = ax[0].imshow(to_plot, cmap='jet')
            
            ax[0].set_title('Power ' + freq_title)
            plt.colorbar(im, ax=ax[0])

            im = ax[1].imshow(phase, vmin=-np.pi, vmax=np.pi, cmap=plt.cm.hsv)
            ax[1].set_title('Phase ' + freq_title)
            plt.colorbar(im, ax=ax[1])
            plt.savefig(join(output_dir, preflix + '_range_' + freq_title + '.png'), dpi=300)
            plt.close()
    
    with open(join(output_dir, preflix + '_peaks.txt'), 'w') as f:
        for item in peak_freq:
            f.write("%s\n" % item)
            
    return 





def main_worker_m(output_dir, no_files, npy_folder, target_freq, preflix, cut_off):
    try:
        mkdir(output_dir)
    except:
        pass


    main_m(no_files, npy_folder, output_dir, target_freq, preflix, cut_off)
    return 


def main_worker_cropped(i):
    output_dir = f"output/{i}_cropped/"
    try:
        mkdir(output_dir)
    except:
        pass
    
    no_files = 15000
    npy_folder = f"npy/{i}_cropped/"
    
    target_freq = 0.4e9
    preflix = 'mediator'

    main_m(no_files, npy_folder, output_dir, target_freq, preflix)
    return 


def main_worker_demag(output_dir, no_files, npy_folder, m_npy_folder, target_freq, preflix, cut_off):
    try:
        mkdir(output_dir)
    except:
        pass
    


    main_demag(no_files, npy_folder, m_npy_folder, output_dir, target_freq, preflix, cut_off)
    return 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cores', type=int, default=1)
    parser.add_argument('--no_files', type=int, default=15000)
    parser.add_argument('--lower_index', type=int, default=0)
    parser.add_argument('--upper_index', type=int, default=13)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--sim_name', type=str)
    parser.add_argument('--output_dir', type=str, default="output")
    parser.add_argument('--cut_off', type=float, default=10e9)
    parser.add_argument('--freq_spacing', type=float, default=0.4e9)
    parser.add_argument('--demag', action='store_true')
    parser.add_argument('--cropped_files', action='store_true')
    parser.add_argument('--prefix_file', type=str, default="prefix.txt")
       
    args = parser.parse_args()
    
    npy_dir_base = join(args.data_folder, 'npy')
    output_dir = join(args.data_folder, args.output_dir)


    try:
        mkdir(output_dir)
    except:
        pass

    gap_list = np.arange(args.lower_index, args.upper_index + 1)

    with multiprocess.Pool(args.no_cores) as p:
        p.starmap(main_worker_m, [(join(output_dir, f"{args.sim_name}{i}_m/"), 
                                   args.no_files, 
                                   join(args.data_folder, f"npy/{args.sim_name}{i}_m/"), 
                                #    0.4e9, 
                                    args.freq_spacing,
                                   'm', args.cut_off) for i in gap_list])


        if args.demag:
            p.starmap(main_worker_demag, [(join(output_dir, f"{args.sim_name}{i}_demag/"),
                                        args.no_files,
                                        join(args.data_folder, f"npy/{args.sim_name}{i}_demag/"),
                                        join(args.data_folder, f"npy/{args.sim_name}{i}_m/"),
                                        args.freq_spacing, 
                                        'demag', args.cut_off
                                        ) for i in gap_list])
            
        
        if args.cropped_files:
            with open(join(args.data_folder, args.prefix_file), 'r') as f:
                prefix_list = f.read().splitlines()
                    
            print(prefix_list)
            for prefix in prefix_list:
                p.starmap(main_worker_m, [(join(output_dir, f"{args.sim_name}{i}_{prefix}_cropped/"),
                                        args.no_files,
                                        join(args.data_folder, f"npy/{args.sim_name}{i}_{prefix}_cropped/"),
                                       args.freq_spacing,
                                         'm', args.cut_off
                                            ) for i in gap_list])
                       
