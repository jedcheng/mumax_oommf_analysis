import pickle
from matplotlib import pyplot as plt
import numpy as np

from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from os.path import join
import multiprocess
import tqdm

def read_pkl_m(pickle_name):

    pickleArray = open(pickle_name, 'rb')
    unpickled = np.load(pickleArray, allow_pickle = True)

    array = unpickled[0]

    x = array[:,:,1,0]
    y = array[:,:,1,1]
    z = array[:,:,1,2]
    
    time = unpickled[1]['SimTime']
    
    return x, y, z, time


def read_data_m(input_path, no_files, 
            preflix, no_cores=8):
    
    first_files = read_pkl_m(join(input_path, preflix + '000000.pkl'))
    
    x0, y0, z0, _ = first_files
    M = np.sqrt(x0**2 + y0**2 + z0**2)
    
    vaccum = np.equal(M, 0)
    shape = first_files[0].shape


    # run in parallel
    def worker(i):
        file_name = preflix + str(i).zfill(6) + '.pkl'
        results = read_pkl_m(join(input_path, file_name))
        
        x_ = results[0] - x0
        y_ = results[1] - y0
        z_ = results[2] - z0
        
        return x_, y_, z_, results[3]
    
    with multiprocess.Pool(no_cores) as p:
        results = p.map(worker, range(no_files))
    p.close()
        
    x = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    z = np.array([r[2] for r in results])
    t = np.array([r[3] for r in results])
        
    return x, y, z, t, shape, vaccum
   
        
def fft_function_m(x, y, z, t, shape, no_files, target_freq):
        


        x_flat = x.reshape(no_files, shape[0]*shape[1]).T
        y_flat = y.reshape(no_files, shape[0]*shape[1]).T
        z_flat = z.reshape(no_files, shape[0]*shape[1]).T
        
        x_fft = np.zeros((shape[0]*shape[1], no_files), dtype=complex)
        y_fft = np.zeros((shape[0]*shape[1], no_files), dtype=complex)
        z_fft = np.zeros((shape[0]*shape[1], no_files), dtype=complex)
    
        for i in range(shape[0]*shape[1]):
            x_fft[i] = fft(x_flat[i])
            y_fft[i] = fft(y_flat[i])
            z_fft[i] = fft(z_flat[i])
        
        # run in parallel
        def worker(i):
            x_fft_ = fft(x_flat[i])
            y_fft_ = fft(y_flat[i])
            z_fft_ = fft(z_flat[i])
            return x_fft_, y_fft_, z_fft_
            
        # with multiprocess.Pool(8) as p:
        #     results = p.map(worker, range(shape[0]*shape[1]))
            
        # p.close()
            
        # for i in range(shape[0]*shape[1]):
        #     x_fft[i] = results[i][0]
        #     y_fft[i] = results[i][1]
        #     z_fft[i] = results[i][2]
            
        xf = fftfreq(no_files, t[1]-t[0])
        fft_spectrum = np.zeros(no_files)
        
        for i in range(no_files):
                fft_spectrum[i] = np.abs(y_fft[:,i]).max() + \
                        np.abs(x_fft[:,i]).max() + np.abs(z_fft[:,i]).max()
    

        distance = int(target_freq*0.8/(xf[1]-xf[0]))
        
        peaks, _ = find_peaks(fft_spectrum, distance=distance)
        
        return xf, fft_spectrum, peaks, x_fft, y_fft, z_fft





def fft_function_m_3D(x, y, z, t, shape, no_files, target_freq):
        
        x_flat = x.reshape(no_files, shape[0]*shape[1]*shape[2]).T
        y_flat = y.reshape(no_files, shape[0]*shape[1]*shape[2]).T
        z_flat = z.reshape(no_files, shape[0]*shape[1]*shape[2]).T
        
        x_fft = np.zeros((shape[0]*shape[1]*shape[2], no_files), dtype=complex)
        y_fft = np.zeros((shape[0]*shape[1]*shape[2], no_files), dtype=complex)
        z_fft = np.zeros((shape[0]*shape[1]*shape[2], no_files), dtype=complex)
    
        for i in range(shape[0]*shape[1]*shape[2]):
            x_fft[i] = fft(x_flat[i])
            y_fft[i] = fft(y_flat[i])
            z_fft[i] = fft(z_flat[i])
        
        # run in parallel
        # def worker(i):
        #     x_fft_ = fft(x_flat[i])
        #     y_fft_ = fft(y_flat[i])
        #     z_fft_ = fft(z_flat[i])
        #     return x_fft_, y_fft_, z_fft_
            
        # with multiprocess.Pool(8) as p:
        #     results = p.map(worker, range(shape[0]*shape[1]))
            
        # p.close()
            
        # for i in range(shape[0]*shape[1]):
        #     x_fft[i] = results[i][0]
        #     y_fft[i] = results[i][1]
        #     z_fft[i] = results[i][2]
            
        xf = fftfreq(no_files, t[1]-t[0])
        fft_spectrum = np.zeros(no_files)
        
        for i in range(no_files):
                fft_spectrum[i] = np.abs(y_fft[:,i]).max() + \
                        np.abs(x_fft[:,i]).max() + np.abs(z_fft[:,i]).max()
    

        distance = int(target_freq*0.8/(xf[1]-xf[0]))
        
        peaks, _ = find_peaks(fft_spectrum, distance=distance)
        
        return xf, fft_spectrum, peaks, x_fft, y_fft, z_fft





def fft_function_m_z(z, t, shape, no_files, target_freq):

        z_flat = z.reshape(no_files, shape[0]*shape[1]).T

        z_fft = np.zeros((shape[0]*shape[1], no_files), dtype=complex)
    
        for i in range(shape[0]*shape[1]):

            z_fft[i] = fft(z_flat[i])
        
        # run in parallel
        # def worker(i):
        #     z_fft_ = fft(z_flat[i])
        #     return z_fft_
            
        # with multiprocess.Pool(8) as p:
        #     results = p.map(worker, range(shape[0]*shape[1]))
            
        # p.close()
            
        # for i in range(shape[0]*shape[1]):
        #     x_fft[i] = results[i][0]
        #     y_fft[i] = results[i][1]
        #     z_fft[i] = results[i][2]
            
        xf = fftfreq(no_files, t[1]-t[0])
        fft_spectrum = np.zeros(no_files)
        
        for i in range(no_files):
                fft_spectrum[i] = np.abs(z_fft[:,i]).max()
    

        distance = int(target_freq*0.8/(xf[1]-xf[0]))
        
        peaks, _ = find_peaks(fft_spectrum, distance=distance)
        
        return xf, fft_spectrum, peaks, z_fft





def read_pkl_demag(pickle_name):

    pickleArray = open(pickle_name, 'rb')
    unpickled = np.load(pickleArray, allow_pickle = True)

    array = unpickled[0]

    x = array[:,:,1,0]
    y = array[:,:,1,1]
    z = array[:,:,1,2]
    
    time = unpickled[1]['SimTime']
    
    return x, y, z, time

def read_data_demag(input_path, no_files, m_preflix, demag_preflix, no_cores=8):
    
    first_files = read_pkl_demag(join(input_path, m_preflix + '000000.pkl'))
    
    x0, y0, z0, _ = first_files
    M = np.sqrt(x0**2 + y0**2 + z0**2)
    
    vaccum = np.equal(M, 0)
    # not_vaccum = np.logical_not(vaccum)
    shape = first_files[0].shape


    # run in parallel
    def worker(i):
        file_name = demag_preflix + str(i).zfill(6) + '.pkl'
        results = read_pkl_demag(join(input_path, file_name))
        
        x_ = results[0] - x0
        y_ = results[1] - y0
        z_ = results[2] - z0
        
        return x_, y_, z_, results[3]
    
    with multiprocess.Pool(no_cores) as p:
        results = p.map(worker, range(no_files))
        
    p.close()
        
    x = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    z = np.array([r[2] for r in results])
    t = np.array([r[3] for r in results])
        
    return x, y, z, t, shape, vaccum
        


def fft_function_demag(x, y, z, t, shape, no_files, target_freq, vaccum):
    
        vaccum_flat = vaccum.reshape(shape[0]*shape[1])
        
        x_flat = x.reshape(no_files, shape[0]*shape[1]).T
        y_flat = y.reshape(no_files, shape[0]*shape[1]).T
        z_flat = z.reshape(no_files, shape[0]*shape[1]).T
        
        x_fft = np.zeros((shape[0]*shape[1], no_files), dtype=complex)
        y_fft = np.zeros((shape[0]*shape[1], no_files), dtype=complex)
        z_fft = np.zeros((shape[0]*shape[1], no_files), dtype=complex)
    
        for i in range(shape[0]*shape[1]):
            x_fft[i] = fft(x_flat[i])
            y_fft[i] = fft(y_flat[i])
            z_fft[i] = fft(z_flat[i])
            
        xf = fftfreq(no_files, t[1]-t[0])
        fft_spectrum = np.zeros(no_files)
        
        for i in range(no_files):
                fft_spectrum[i] = np.abs(y_fft[:,i][vaccum_flat]).max() + \
                        np.abs(x_fft[:,i][vaccum_flat]).max() + np.abs(z_fft[:,i][vaccum_flat]).max()
    

        distance = int(target_freq*0.8/(xf[1]-xf[0]))
        
        peaks, _ = find_peaks(fft_spectrum, distance=distance)
        
        return xf, fft_spectrum, peaks, x_fft, y_fft, z_fft



# import oommfdecode2
# from os.path import join, splitext, basename

# def convert_to_pkl(input_path, prefix, no_files, out_dir, no_cores=8):
    
    
#     def worker(in_file):
#         out_file = in_file
#         array, headers, extra = oommfdecode2.unpackFile(in_file)
#         file_name = splitext(basename(out_file))[0]
#         oommfdecode2.pickleArray(array, headers, extra, join(out_dir, file_name + '.pkl'))

#     pool = multiprocess.Pool(no_cores)
#     files = [join(input_path, f'{prefix}{str(i).zfill(6)}.ovf') for i in range(no_files)]
    
#     for _ in tqdm.tqdm(pool.imap_unordered(worker, files), total=len(files)):
#         pass
              
              
#     pool.close()
#     return 