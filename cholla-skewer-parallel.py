import numpy as np
import argparse
import h5py
from tqdm import tqdm
import time
from multiprocessing import Process, Queue, cpu_count
import queue
import os

#######################################
# Create command line argument parser
#######################################

def create_parser():
    parser = argparse.ArgumentParser(description="Flags and options from user.")

    parser.add_argument('-i', '--input', dest='input', default='data/', type=str,
                        help='Directory with input files to process.')
    parser.add_argument('-o', '--output', dest='output', default='output', type=str,
                        help='Base name for output skewer data (coordinates will be appended).')
    parser.add_argument('--snap', dest='snap', default=1, type=str, help='Snapshot to read.')
    
    parser.add_argument('--ngpux', dest='ngpux', default=16, type=int, help='Subvolumes in x')
    parser.add_argument('--ngpuy', dest='ngpuy', default=16, type=int, help='Subvolumes in y')
    parser.add_argument('--ngpuz', dest='ngpuz', default=16, type=int, help='Subvolumes in z')
    
    parser.add_argument('--nskewers', dest='nskewers', default=1, type=int,
                        help='Number of skewers from subvolume (default: 1)')
    
    parser.add_argument('-x', dest='x', action='store_true', default=False, help='Skewers along x')
    parser.add_argument('-y', dest='y', action='store_true', default=False, help='Skewers along y')
    parser.add_argument('-z', dest='z', action='store_true', default=False, help='Skewers along z')

    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', default=False)
    
    parser.add_argument('--nprocs', dest='nprocs', default=cpu_count(), type=int,
                        help='Number of parallel processes to use')

    return parser


#######################################
# Worker function (Consumer)
#######################################
def worker(task_queue, progress_queue, args):
    """
    Pulls row tasks from the queue continuously until it hits a None (sentinel).
    """
    # Keep reading from the stack until it's empty
    while True:
        task = task_queue.get()
        
        # A None task is the signal to terminate this worker
        if task is None:
            break
            
        ix = task['ix']
        iy = task['iy']
        iz = task['iz']

        if args.z:
            di = 0
            n = args.ngpuy * args.ngpuz * ix + args.ngpuz * iy + np.arange(args.ngpuz)
            mom = 'momentum_z'
            out_file = f"{args.output}_ix{ix}_iy{iy}.h5"
        elif args.y:
            di = 1
            n = args.ngpuy * args.ngpuz * ix + args.ngpuz * np.arange(args.ngpuy) + iz
            mom = 'momentum_y'
            out_file = f"{args.output}_ix{ix}_iz{iz}.h5"
        elif args.x:
            di = 2
            n = args.ngpuy * args.ngpuz * np.arange(args.ngpux) + args.ngpuz * iy + iz
            mom = 'momentum_x'
            out_file = f"{args.output}_iy{iy}_iz{iz}.h5"

        attr_write = ['Current_a', 'Current_z', 'Git Commit Hash', 'H0', 'Omega_L', 'Omega_M', 
                      'density_unit', 'domain', 'dx', 'energy_unit', 'gamma', 'length_unit', 
                      'mass_unit', 'n_step', 't', 'time_unit', 'velocity_unit']

        nskewers = args.nskewers
        sxi = np.zeros(nskewers, dtype=int)
        syi = np.zeros(nskewers, dtype=int)

        s_HI, s_T, s_v = None, None, None
        attrw = {}

        for i, skidx in enumerate(n):
            fname = f'{args.input}/{args.snap}/{args.snap}.h5.{skidx}'
            
            with h5py.File(fname, 'r') as f:
                if i == 0:
                    ns = f.attrs['dims'][di]
                    sxi = np.random.randint(f.attrs['dims_local'][di], size=nskewers)
                    syi = np.random.randint(f.attrs['dims_local'][di], size=nskewers) 

                    for attr in attr_write:
                        if attr in f.attrs:
                            attrw[attr] = f.attrs[attr]

                    s_HI = np.zeros((nskewers, ns))
                    s_T  = np.zeros((nskewers, ns))
                    s_v  = np.zeros((nskewers, ns))

                nsl    = f.attrs['dims_local'][di]
                offset = f.attrs['offset'][di]

                for j in range(nskewers):
                    if args.x:
                        slice_x = slice(0, nsl, 1)
                        slice_y = sxi[j]
                        slice_z = syi[j]
                    elif args.y:
                        slice_y = slice(0, nsl, 1)
                        slice_x = syi[j]
                        slice_z = sxi[j]
                    elif args.z:
                        slice_z = slice(0, nsl, 1)
                        slice_y = syi[j]
                        slice_x = sxi[j]

                    s_HI[j, offset:offset+nsl] = np.asarray(f['HI_density'][slice_x, slice_y, slice_z]).flatten()
                    s_T[j, offset:offset+nsl]  = np.asarray(f['temperature'][slice_x, slice_y, slice_z]).flatten()
                    s_v[j, offset:offset+nsl]  = np.asarray(f[mom][slice_x, slice_y, slice_z] / f['density'][slice_x, slice_y, slice_z]).flatten()

        # Write out the HDF5 results for this row
        with h5py.File(out_file, 'w') as g:
            for attr in attr_write:
                if attr in attrw:
                    g.attrs[attr] = attrw[attr]
            g.create_dataset('HI_density', data=s_HI)
            g.create_dataset('temperature', data=s_T)
            g.create_dataset('velocity', data=s_v)
            
        # Signal that one row has been completed
        progress_queue.put(1)


#######################################
# main() function (Producer)
#######################################
def main():
    time_global_start = time.time()

    parser = create_parser()
    args   = parser.parse_args()

    if not (args.x or args.y or args.z):
        print('Must choose a direction (-x, -y, -z)')
        exit(0)

    # 1. Initialize the stack (queue)
    task_queue = Queue()
    progress_queue = Queue()
    total_tasks = 0

    # 2. Populate the queue with all possible rows
    if args.z:
        for ix in range(args.ngpux):
            for iy in range(args.ngpuy):
                task_queue.put({'ix': ix, 'iy': iy, 'iz': 0})
                total_tasks += 1
    elif args.y:
        for ix in range(args.ngpux):
            for iz in range(args.ngpuz):
                task_queue.put({'ix': ix, 'iy': 0, 'iz': iz})
                total_tasks += 1
    elif args.x:
        for iy in range(args.ngpuy):
            for iz in range(args.ngpuz):
                task_queue.put({'ix': 0, 'iy': iy, 'iz': iz})
                total_tasks += 1

    # Add sentinels (None) to tell the workers when to exit. 
    # We need one sentinel for every process we spawn.
    for _ in range(args.nprocs):
        task_queue.put(None)

    if args.verbose:
        print(f"Total rows placed in queue: {total_tasks}")
        print(f"Spinning up {args.nprocs} processes...")

    # 3. Start the worker processes
    processes = []
    for _ in range(args.nprocs):
        p = Process(target=worker, args=(task_queue, progress_queue, args))
        p.start()
        processes.append(p)

    # 4. Monitor progress from the main thread
    # The main thread blocks here, waiting for workers to push '1' to the progress_queue
    for _ in tqdm(range(total_tasks)):
        progress_queue.get()

    # 5. Clean up processes
    for p in processes:
        p.join()

    time_global_end = time.time()
    if args.verbose:
        print(f"Time to execute program: {time_global_end - time_global_start:.2f}s.")

#######################################
# Run the program
#######################################
if __name__=="__main__":
    main()