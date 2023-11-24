import struct
import numpy as np

def omf_decode(filename):
    f = open(filename, "rb")

    while True:
        line = f.readline()
        # print(line)


        # if line.startswith(b'# Desc:  Total simulation time'):
        #     t = float(line.split()[-2])
        #     print(t)
        if "Total simulation time" in str(line):
            t = float(line.split()[-2])
            # print(t)


        elif line.startswith(b'# xnodes:'):
            xnodes = int(line.split()[-1])
            # print(xnodes)

            line = f.readline()
            ynodes = int(line.split()[-1])
            # print(ynodes)

            line = f.readline()
            znodes = int(line.split()[-1])
            # print(znodes)

            output_array = np.zeros((xnodes, ynodes, znodes, 3))
        elif line.startswith(b'# Begin: Data Binary'):
            # # Begin: Data Binary 4

            binary = int(line.split()[-1])

            break

    
    if binary == 8:
    

        flag = f.read(8)

        if struct.unpack("<d", flag)[0] != 123456789012345.0:
            raise ValueError("Wrong flag")
        

        for k in range(znodes):
            for j in range(ynodes):
                for i in range(xnodes):
                    for l in range(3):
                        output_array[i, j, k, l] = struct.unpack("<d", f.read(8))[0]


        return output_array, t
    

    elif binary == 4:

        flag = f.read(4)

        if struct.unpack("<f", flag)[0] != 1234567.0:
            raise ValueError("Wrong flag")
        
        # for i in range(xnodes):
        #     for j in range(ynodes):
        #         for k in range(znodes):
        #             for l in range(3):
        #                 output_array[i, j, k, l] = struct.unpack("<f", f.read(4))[0]

        for k in range(znodes):
            for j in range(ynodes):
                for i in range(xnodes):
                    for l in range(3):
                        output_array[i, j, k, l] = struct.unpack("<f", f.read(4))[0]



        return output_array, t
    
    else:
        raise ValueError("Wrong binary")
    

    