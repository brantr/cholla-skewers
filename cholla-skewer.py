import numpy as np
import matplotlib.pyplot as plt
import argparse
import h5py
from tqdm import tqdm
import time

#######################################
# Create command line argument parser
#######################################

def create_parser():

	# Handle user input with argparse
    parser = argparse.ArgumentParser(
        description="Flags and options from user.")

    parser.add_argument('-i', '--input',
        dest='input',
        default='data/',
        metavar='input',
        type=str,
        help='Directory with input files to process.')

    parser.add_argument('-o', '--output',
        dest='output',
        default='output.h5',
        metavar='output',
        type=str,
        help='Output skewer data.')

    parser.add_argument('--snap',
        dest='snap',
        default=1,
        metavar='snap',
        type=str,
        help='Snapshot to read.')

    parser.add_argument('--ngpux',
        dest='ngpux',
        default=16,
        metavar='ngpux',
        type=int,
        help='Number of subvolumes in x direction')

    parser.add_argument('--ngpuy',
        dest='ngpuy',
        default=16,
        metavar='ngpuy',
        type=int,
        help='Number of subvolumes in y direction')

    parser.add_argument('--ngpuz',
        dest='ngpuz',
        default=16,
        metavar='ngpuz',
        type=int,
        help='Number of subvolumes in z direction')

    parser.add_argument('--ix',
        dest='ix',
        default=0,
        metavar='ix',
        type=int,
        help='Subvolume in x direction')

    parser.add_argument('--iy',
        dest='iy',
        default=0,
        metavar='iy',
        type=int,
        help='Subvolume in y direction')

    parser.add_argument('--iz',
        dest='iz',
        default=0,
        metavar='iz',
        type=int,
        help='Subvolume in z direction')

    parser.add_argument('--nskewers',
        dest='nskewers',
        default=1,
        metavar='nskewers',
        type=int,
        help='Number of skewers from subvolume (default: 1)')

    parser.add_argument('-x',
        dest='x',
        action='store_true',
        help='Skewers along x direction? (default: False)',
        default=False)
    parser.add_argument('-y',
        dest='y',
        action='store_true',
        help='Skewers along y direction? (default: False)',
        default=False)

    parser.add_argument('-z',
        dest='z',
        action='store_true',
        help='Skewers along z direction? (default: False)',
        default=False)


    parser.add_argument('-v', '--verbose',
        dest='verbose',
        action='store_true',
        help='Print helpful information to the screen? (default: False)',
        default=False)

    return parser

#######################################
# main() function
#######################################
def main():

    #begin timer
    time_global_start = time.time()

    #create the command line argument parser
    parser = create_parser()

    #store the command line arguments
    args   = parser.parse_args()


    # The gpu subvolumes are tiled in z, y, x order

    ix = args.ix
    iy = args.iy
    iz = args.iz

    ngpux = args.ngpux
    ngpuy = args.ngpuy
    ngpuz = args.ngpuz



    # Determine the files
    # needed to load
    if(args.z):
        di = 0
        n = ngpuy*ngpuz*ix + ngpuz*iy + np.arange(ngpuz)
        mom = 'momentum_z'
    if(args.y):
        di = 1
        n = ngpuy*ngpuz*ix + ngpuz*np.arange(ngpuy) + iz
        mom = 'momentum_y'
    if(args.x):
        di = 2
        n = ngpuy*ngpuz*np.arange(ngpux) + ngpuz*iy + iz
        mom = 'momentum_x'

    if((not args.x) and (not args.y) and (not args.z)):
        print(f'Must choose a direction (-x,-y,-z)')
        exit(0)


    # which attributes to keep?
    #attr_write = ['Current_a', 'Current_z', 'Git Commit Hash', 'H0', 'Omega_L', 'Omega_M', 'Omega_B', 'Omega_K', 'density_unit', 'domain', 'dx', 'energy_unit', 'gamma', 'length_unit', 'mass_unit', 'n_step', 't', 'time_unit', 'velocity_unit']
    attr_write = ['Current_a', 'Current_z', 'Git Commit Hash', 'H0', 'Omega_L', 'Omega_M', 'density_unit', 'domain', 'dx', 'energy_unit', 'gamma', 'length_unit', 'mass_unit', 'n_step', 't', 'time_unit', 'velocity_unit']

    nskewers = args.nskewers
    sxi = np.zeros(nskewers)
    syi = np.zeros(nskewers)


    # generate the skewers

    for i, skidx in enumerate(tqdm(n)):

        # load a single file for info
        fname = f'{args.input}/{args.snap}/{args.snap}.h5.{i}'

        print(fname)

        f = h5py.File(fname,'r')

        if(i == n[0]):

            #for key in f.attrs.keys():
            #    print(key)
            #exit(0)

            # get the skewer lengths
            ns = f.attrs['dims'][di]

            sxi = np.random.randint(f.attrs['dims_local'][di],size=nskewers)
            syi = np.random.randint(f.attrs['dims_local'][di],size=nskewers) 


            # record the attributes to keep
            attrw = {}
            for attr in attr_write:
                attrw[attr] = f.attrs[attr]

            s_HI = np.zeros((nskewers,ns))
            s_T  = np.zeros((nskewers,ns))
            s_v  = np.zeros((nskewers,ns))

        nsl    = f.attrs['dims_local'][di]
        offset = f.attrs['offset'][di]

        print(f'i {i} offset {offset}')

        for j in range(nskewers):
            if(args.x):
                slice_x = slice(0,nsl,1)
                slice_y = sxi[j]
                slice_z = syi[j]
            if(args.y):
                slice_y = slice(0,nsl,1)
                slice_x = syi[j]
                slice_z = sxi[j]
            if(args.z):
                slice_z = slice(0,nsl,1)
                slice_y = syi[j]
                slice_x = sxi[j]


            s_HI[j,offset:offset+nsl] = np.asarray(f['HI_density'][slice_x,slice_y,slice_z]).flatten()
            s_T[j,offset:offset+nsl]  = np.asarray(f['temperature'][slice_x,slice_y,slice_z]).flatten()
            s_v[j,offset:offset+nsl]  = np.asarray(f[mom][slice_x,slice_y,slice_z]/f['density'][slice_x,slice_y,slice_z]).flatten()



    # write out the HDF5 results
    with h5py.File(args.output,'w') as g:
        for attr in attr_write:
            g.attrs[attr] = attrw[attr]
        g.create_dataset('HI_density',data=s_HI)
        g.create_dataset('temperature',data=s_T)
        g.create_dataset('velocity',data=s_v)

    #end timer
    time_global_end = time.time()
    if(args.verbose):
    	print(f"Time to execute program: {time_global_end-time_global_start}s.")

#######################################
# Run the program
#######################################
if __name__=="__main__":
	main()
