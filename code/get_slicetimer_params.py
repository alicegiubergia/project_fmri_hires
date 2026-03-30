import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

#Sequence parameters


parser = argparse.ArgumentParser(prog='slicetimer_params',
                                 description='Estimate slicetimer parameters for slicetimer correction.')
parser.add_argument('-out', metavar='FOLDER', type=str, required=True,
                    help='Output folder.')
parser.add_argument('-n_slices', metavar='int', type=int, required=True,
                    help='Number of slices.')
parser.add_argument('-tpattern', metavar='string', type=str, required=True, choices = ['FH', 'HF', 'interleaved'],
                    help='Pattern of slice acquisition.')
parser.add_argument('-MB', metavar='int', type=str, required=True,
                    help='Multi-Band Factor.')

#parser.print_help()
args = parser.parse_args()


print(parser.prog)
print("\n")
print(parser.description)
print("---------------")
print("Input arguments")
print("---------------")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
    
def my_func(row):
    t = 0.5 - bin/2 - (row) * bin
    return t

Seq_path=args.out
Slices=args.n_slices
Slice_scan_order=args.tpattern
MB_SENSE_factor=args.MB

'''
if MB_SENSE_factor == 1:
    MB_SENSE='no'
    MB_SENSE_factor=1
else:
    MB_SENSE='yes'
    MB_SENSE_factor = int(MB_SENSE_factor)
'''

MB_SENSE_factor = int(MB_SENSE_factor)
SlTiFilename="slice_timings.txt"

Seq_output_path = Seq_path # Seq_path.replace('dataset_test', 'dataset_test_fMRIpreproc')

print("\nWriting Slice Timings txt file for FSL's slicetimer ... ")

Slices_per_band = Slices // MB_SENSE_factor

bin = round((1/Slices_per_band), 3)
#print('bin is %f' %bin)

timings = [] 

if Slice_scan_order=='HF':
	timings=-1*timings
if Slice_scan_order=='interleaved':
    if (MB_SENSE_factor>1):
        slice_indices = np.arange(Slices)
        interleaved_order = np.zeros(Slices, dtype=int)

        # Create interleaved pattern
        even_idx = slice_indices[::2]
        odd_idx = slice_indices[1::2]
        interleaved_pattern = np.concatenate((even_idx, odd_idx))

        for mb in range(MB_SENSE_factor):
            band_indices = interleaved_pattern[mb::MB_SENSE_factor]
            Slice_order = pd.Series(np.arange(len(band_indices)), name='order')
            t_band = Slice_order.apply(my_func)
            t_band = t_band.apply(lambda x: round(x, 3))

            temp = pd.Series(index=band_indices, data=t_band.values)
            timings.append(temp)

        timings = pd.concat(timings).sort_index()

    elif MB_SENSE_factor==1:
        SliceGroup=0
        Slice_order=pd.Series(np.arange(0,Slices), name='order')
        t_band = Slice_order.apply(my_func)
        step=round(np.sqrt(Slices))
        for k in np.arange(1,Slices):
            current=Slice_order.iloc[k-1]+step
            #print(current)
            if current>Slices-1:
                SliceGroup=SliceGroup+1;
                current=SliceGroup;
            Slice_order.iloc[k]=current
        timings = t_band.iloc[Slice_order]
        print("\nTimings\n")
        print(timings)
           
                


fig, ax = plt.subplots(figsize=(15,5))
plt.stem(timings.reset_index().index.to_numpy(), timings.to_numpy())
plt.xticks(timings.reset_index().index)
ax.set_xticklabels(timings.reset_index().index+1)
plt.grid()
plt.ylabel('Shifting in fraction of TR')
plt.xlabel('Slices')
plt.title('Slice Timings, NSlices: ' + str(Slices) + ' MB factor: '+str(MB_SENSE_factor))

plt.savefig(os.path.join(Seq_output_path, SlTiFilename.split('.')[0] + '.png'), dpi=400)
timings.to_csv(os.path.join(Seq_output_path, SlTiFilename), sep='\n', index=False, header=False)

if  os.path.exists(os.path.join(Seq_output_path, SlTiFilename)):
	print("\nFile %s created." % str(os.path.join(Seq_output_path, SlTiFilename)))
