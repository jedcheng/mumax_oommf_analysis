# import oommfdecode2

from omf_decode import omf_decode
import multiprocess
import tqdm

from os.path import join, splitext, basename

import numpy as np

import os 
import argparse

from glob import glob 





def read_data_m(input_path, no_files, preflix, no_cores=8):
    
    
    
    
    # first_file, _, _ = oommfdecode2.unpackFile(join(input_path, preflix + '000000.ovf'))
    first_file, _ = omf_decode(join(input_path, preflix + '000000.ovf'))
    
    
    x0, y0, z0 = first_file[:,:,:,0], first_file[:,:,:,1], first_file[:,:,:,2]
    M = np.sqrt(x0**2 + y0**2 + z0**2)
    
    vaccum = np.equal(M, 0)
    shape = first_file[:,:,:,0].shape


    # run in parallel
    def worker(i):
        file_name = preflix + str(i).zfill(6) + '.ovf'
        # array, _, extra = oommfdecode2.unpackFile(join(input_path, file_name))
        array, t_ = omf_decode(join(input_path, file_name))

        x_ = array[:,:,:,0] - x0
        y_ = array[:,:,:,1] - y0
        z_ = array[:,:,:,2] - z0
        
        # t_ = extra['SimTime']
        
        return x_, y_, z_, t_


    with multiprocess.Pool(no_cores) as p:
        results = list(tqdm.tqdm(p.imap(worker, range(no_files)), total=no_files))
        
        
    x = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    z = np.array([r[2] for r in results])
    t = np.array([r[3] for r in results])
        
    return x, y, z, t, shape, vaccum






def main(no_files, output_dir, input_dir,
         preflix, no_cores):
    
    x, y, z, t, shape, vaccum = read_data_m(input_dir, no_files, preflix, no_cores)
    
    np.save(join(output_dir, 'x.npy'), x)
    np.save(join(output_dir, 'y.npy'), y)
    np.save(join(output_dir, 'z.npy'), z)
    np.save(join(output_dir, 't.npy'), t)
    
    np.save(join(output_dir, 'shape.npy'), shape)
    np.save(join(output_dir, 'vaccum.npy'), vaccum)
    
    return 






    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert OOMMF .ovf files to .pkl files then to .npy files')
    parser.add_argument('--no_files', type=int)
    parser.add_argument('--no_cores', type=int)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--sim_name', type=str)
    parser.add_argument('--demag', action='store_true')
    parser.add_argument('--cropped_files', action='store_true')
    parser.add_argument('--prefix_file', type=str, default="prefix.txt")
             
    args = parser.parse_args()
    
    # index = args.index
    sim_name = args.sim_name
    

    input_path = join(args.data_folder, sim_name + ".out")
    output_dir_base = join(args.data_folder, 'npy', sim_name)                        
                             
    

    m_folder = output_dir_base + "_m"
    try:
        os.makedirs(m_folder)
    except:
        pass
    
    
    print(f"Processing M for {sim_name}")
    
    main(args.no_files, m_folder, 
         input_path,
         "m", args.no_cores)
    
    


    if args.demag:
        print(f"Processing H for {sim_name}")
        demag_folder = output_dir_base + "_demag"
        try:
            os.makedirs(demag_folder)
        except:
            pass 
        main(args.no_files, demag_folder, 
                input_path,
                "B_demag", args.no_cores)
        
        
    
    
    
    if args.cropped_files:
        
        with open(join(args.data_folder, args.prefix_file), 'r') as f:
            prefix_list = f.read().splitlines()
        print(prefix_list)
        for index, prefix in enumerate(prefix_list):
            
            print(f"Processing {prefix} for {index}")
            # cropped =  prefix + "_cropped"
            cropped = output_dir_base + "_" + prefix + "_cropped"
            
            try:
                os.makedirs(cropped)
            except:
                pass
            
            main(args.no_files, cropped,
                    input_path,
                    prefix, args.no_cores)
                    

