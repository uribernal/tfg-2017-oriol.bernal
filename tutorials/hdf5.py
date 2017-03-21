import numpy as np
import h5py

matrix1 = np.random.random(size = (1000,1000))
matrix2 = np.random.random(size = (1000,1000))
matrix3 = np.random.random(size = (1000,1000))
matrix4 = np.random.random(size = (1000,1000))

with h5py.File('/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/hdf5_data.h5', 'w') as hdf:
    G1 = hdf.create_group('Group1')
    G1.create_dataset('dataset1', data=matrix1, compression='gzip', compression_opts=9)
    G1.create_dataset('dataset4', data=matrix4, compression='gzip', compression_opts=9)

    G21 = hdf.create_group('Group2/SubGroup1')
    G21.create_dataset('dataset3', data=matrix3, compression='gzip', compression_opts=9)

    G22 = hdf.create_group('Group2/SubGroup2')
    G22.create_dataset('dataset2', data=matrix2, compression='gzip', compression_opts=9)

#du -sh hdf5_data.h5 to check size


with h5py.File('/home/uribernal/Desktop/MediaEval2016/devset/continuous-movies/DB/hdf5_data.h5', 'r') as hdf:
    ls = list(hdf.keys())
    print('List of datasets in this file: \n', ls)
    data = hdf.get('dataset1')
    dataset1 = np.array(data)
    print('Shape of the dataset: \n', dataset1.shape)

    base_items = list(hdf.items())
    print('Items in the base directory: \n', base_items)
    G1 = hdf.get('Group1')
    G1_items = list(G1.items())
    print('Items in Group1: \n', G1_items)
    dataset4 = np.array(G1.get('dataset4'))
    print(dataset4.shape)

    G2 = hdf.get('Group2')
    G2_items = list(G2.items())
    print('Items in Group2:', G2_items)
    G21 = G2.get('/Group2/SubGroup1')
    G21_items = list(G21.items())
    print('Items in Group21:', G21_items)
    dataset3 = np.array(G21.get('dataset3'))
    print(dataset3.shape)
