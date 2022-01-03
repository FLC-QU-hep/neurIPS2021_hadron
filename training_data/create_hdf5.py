import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import uproot
import logging
import os, sys
import argparse
import multiprocessing as mp
import pickle


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

## function for pixels
def pixs (strt,end):

    if hcal and ecal:
        nlayers = 78
    elif hcal:
        nlayers = 48
        binX = np.arange(-740, 741, 30.5)
        binZ = np.arange(271, 1751, 30.5)
    else:
        nlayers = 30  
        binX = np.arange(-81, 82, 5.088333)
        binZ = np.arange(-77, 78, 5.088333)

    ## Temporary storage for numpy arrays (layer information)
    l = []
    cell_map = {
        (0, 0, 0): [0.0, 0.0, 0.0, 0.0, 0.0 ]
    }

    for i in range(strt, end):
        fig, axs = plt.subplots(nlayers, 1, figsize=(30, 20))

        layers = []
       

        for j in range(0,nlayers):
            if hcal:
                idx = np.where((y[i] <= (hmap[j] + 5.0)) & (y[i] > (hmap[j] - 5.0)))
            elif ecal: 
                idx = np.where((y[i] <= (hmap[j] + 0.9999)) & (y[i] > (hmap[j] + 0.0001)))
            
            xlayer = x[i].take(idx)[0]
            zlayer = z[i].take(idx)[0]
            elayer = e[i].take(idx)[0]
            cid0 = id0[i].take(idx)[0]
            cid1 = id1[i].take(idx)[0]

            ### GeV -- > MeV conversion for cell energies
            elayer = elayer * 1000.00

            ### 2d hist is need for energy weighted distributions
            h0 = axs[j].hist2d(xlayer, zlayer, bins=[binX, binZ], weights=elayer)
            layers.append(h0[0])

            if opt.cellmap:
                non_x, non_z = np.nonzero(h0[0]) 
                for k in range(0,len(non_x)):
                    xval = h0[1][non_x[k]]
                    zval = h0[2][non_z[k]]
                    x_real, idX = find_nearest(xlayer, xval)
                    z_real, idZ = find_nearest(zlayer, zval)
                    
                    cell_map[(j, non_x[k], non_z[k])] =  [x_real, hmap[j], z_real, cid0.take(idX), cid1.take(idX)]
            



        ## accumulate for each event
        l.append(layers)
        plt.close(fig)

    layers = np.asarray(l)

    if opt.cellmap:
        return cell_map
    else:
        return layers

def E0(strt,end):
    ## get in incident energy
    e0 = []
    for i in range(strt, end):
        tmp = np.reshape(mcEne[i].take([0]), (1,1))
        e0.append(tmp)

    e0 = np.reshape(np.asarray(e0),(-1,1))
    return e0

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, help='number of cpus', default=1)
    parser.add_argument('--rootfile', type=str, required=True, help='input root file for streaming')
    parser.add_argument('--branch', type=str, required=True, help='branch name of the root file')
    parser.add_argument('--batchsize', type=int, help='batch size for streaming', default=100)
    parser.add_argument('--hcal', type=bool, help='include HCAL', default=False)
    parser.add_argument('--ecal', type=bool, help='include ECAL', default=False)
    parser.add_argument('--output', type=str, required=True, help='output hdf5 file')
    parser.add_argument('--cellmap', type=bool, help='creates pickle file for cell_id maps', default=False)


    opt = parser.parse_args()

    ncpu = int(opt.ncpu)
    root_path = str(opt.rootfile)
    root_branch = str(opt.branch)
    batch = int(opt.batchsize)
    hcal = opt.hcal
    ecal = opt.ecal
    out = str(opt.output)

    if hcal and ecal:
        ## create hit map of HCAL + ECAL(in y coordinate[cm])
        hmap = np.array([1811, 1814, 1824, 1827, 1836, 1839, 1849,
                    1852, 1861, 1864, 1873, 1877, 1886, 1889, 1898, 1902,
                    1911, 1914, 1923, 1926, 1938, 1943, 1955, 1960,
                    1971, 1976, 1988, 1993, 2005, 2010, ### HCAL starts
                    2079, 2108, 2134, 2160, 2187, 2213, 2239, 2265, 2294, 2320, 2346,
                    2372, 2398, 2424, 2450, 2479, 2506, 2532, 2558, 2584, 2610, 2636,
                    2662, 2691, 2717, 2743, 2769, 2796, 2822, 2848, 2877, 2903, 2929,
                    2955, 2981, 3007, 3033, 3062, 3088, 3115, 3141, 3167, 3193, 3219,
                    3248, 3274, 3300, 3326
       ])


    elif hcal:
         hmap = np.array([
                    2079, 2108, 2134, 2160, 2187, 2213, 2239, 2265, 2294, 2320, 2346,
                    2372, 2398, 2424, 2450, 2479, 2506, 2532, 2558, 2584, 2610, 2636,
                    2662, 2691, 2717, 2743, 2769, 2796, 2822, 2848, 2877, 2903, 2929,
                    2955, 2981, 3007, 3033, 3062, 3088, 3115, 3141, 3167, 3193, 3219,
                    3248, 3274, 3300, 3326
         ])

    else:
        ## create hit map of ECAL (in y coordinate[cm])
        hmap = np.array([1811, 1814, 1824, 1827, 1836, 1839, 1849,
                    1852, 1861, 1864, 1873, 1877, 1886, 1889, 1898, 1902,
                    1911, 1914, 1923, 1926, 1938, 1943, 1955, 1960,
                    1971, 1976, 1988, 1993, 2005, 2010])


    #stream from root file
    ntuple = uproot.open(root_path)[root_branch]
    x = ntuple.array("scpox")
    y = ntuple.array("scpoy")
    z = ntuple.array("scpoz")
    e = ntuple.array("scene")
    id0 = ntuple.array("scci0")
    id1 = ntuple.array("scci1")
    mcPDG = ntuple.array("mcpdg")
    mcEne = ntuple.array("mcene")

    import time
    start_time = time.time()

    
    
    if opt.cellmap:
        #cmaps = pool.starmap(pixs, events)
        print("Creating cell-id map. Single CPU job")
        cmaps = pixs(0,20000)
    else:
        ## execute cpu jobs
        print("Creating {}-process pool".format(ncpu) )
        pool = mp.Pool(ncpu)
        evts = np.arange(0, x.shape[0], batch)
        #evts = np.arange(0, 65001, batch)
        tmp = [[evts[k-1],evts[k]] for k in range(1,len(evts))]
        events = np.vstack(tmp)

        pixels = pool.starmap(pixs, events)
        e0 = pool.starmap(E0, events)

    

    #Open HDF5 file for writing
    if not opt.cellmap:
        hf = h5py.File(out + '.hdf5', 'w')
        if hcal and ecal:
            grp = hf.create_group("ecal_hcal")
        elif hcal:
            grp = hf.create_group("hcal_only")
        else :
            grp = hf.create_group("ecal")

        ## write to hdf5 files
        grp.create_dataset('layers', data=np.vstack(pixels))
        grp.create_dataset('energy', data=np.vstack(e0))

        pool.close()
        pool.join()

    else:  ## create mapping (via pickle file)
        with open('cell-map_HCAL.pickle', 'wb') as fp:
            pickle.dump(cmaps, fp, protocol=pickle.HIGHEST_PROTOCOL)



    print("--- %s seconds ---" % (time.time() - start_time))
