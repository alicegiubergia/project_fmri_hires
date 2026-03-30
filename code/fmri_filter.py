#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 10:34:32 2023

@author: agiubergia
"""

import os
import argparse
import numpy as np
import nibabel as nib
import nibabel.processing 
import nibabel.nifti1 as nifti
import nilearn.signal
import scipy.signal

parser = argparse.ArgumentParser(description='Filter fMRI data.')
parser.add_argument('-epi', metavar='FILE', type=str, required=True,
                    help='EPI file.')
parser.add_argument('-mas', metavar='FILE', type=str, required=True,
                    help='MASK file in EPI space.')
parser.add_argument('-TR', metavar='N', type=float, required=True,
                    help='Repetition Time.')
parser.add_argument('-hp', metavar='N', required=False,
                    action='store', type=float,
                    help='Cut-off frequency for High-Pass Filtering in Hz.')
parser.add_argument('-lp', metavar='N', required=False,
                    action='store', type=float, 
                    help='Cut-off frequency for Low-Pass Filtering in Hz.')
parser.add_argument('-out', metavar='OUTPUT EPI REGRESSED FILE', type=str, required=True,
                    help='Output file.')
#parser.print_help()
args = parser.parse_args()

print(parser.description)
print("---------------")
print("Input arguments")
print("---------------")

for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
    
# Check input files
ext = os.path.splitext(args.epi)[1]
if not ext or (ext != '.nii' and ext != '.gz'):
    print('EPI not valid, add adequate file extension to EPI variable...')
ext = os.path.splitext(args.mas)[1]
if not ext or (ext != '.nii' and ext != '.gz'):
    print('MASK not valid, add adequate file extension to MASK variable...')

# Load fMRI and mask        
EPIimg=nib.load(args.epi)
EPIdata=EPIimg.get_fdata()
EPIhdr=EPIimg.header
EPIaffine=EPIimg.affine

MASKimg=nib.load(args.mas)
MASKdata=MASKimg.get_fdata()
MASKdata=MASKdata.astype(bool)
MASKhdr=MASKimg.header
MASKaffine=MASKimg.affine

reshEPI = EPIdata.reshape(EPIdata.shape[0]*EPIdata.shape[1]*EPIdata.shape[2],EPIdata.shape[3])
reshMASK = MASKdata.reshape(MASKdata.shape[0]*MASKdata.shape[1]*MASKdata.shape[2])
filt_EPI = np.zeros(reshEPI.shape)

print("\nTemporal demeaning")
EPIm = np.mean(EPIdata, axis=-1)
EPIdata = EPIdata - EPIm[:, :, :, np.newaxis]
n=0
Nvol=EPIdata.shape[3]

print(f"\nApplying a high-pass Butterworth filter to {args.epi}")
for i, tc in enumerate(reshEPI):
    if reshMASK[i]:
        
        filt_EPI[i] =  nilearn.signal.butterworth(signals=tc, 
                                          sampling_rate=1/args.TR, 
                                          low_pass=None,
                                          high_pass=args.hp,
                                          order=5)
    else:
        filt_EPI[i] = 0

if args.lp:
    n=0
    print(f"\nApplying a low-pass Chebyshev filter to {args.epi}")
    rs_dB = -20*np.log10(0.05)
    for i, tc in enumerate(filt_EPI):   
        if reshMASK[i]:
            #print(".", end='')
            b,a = scipy.signal.cheby2(N=6,
                                      rs=rs_dB,
                                      Wn=args.lp,
                                      btype='low',
                                      fs=1/args.TR)
            filt_EPI[i] = scipy.signal.filtfilt(b, a, tc)
        else:
            filt_EPI[i] = 0

# Save results
filt_EPI = filt_EPI.reshape(EPIimg.shape[0], EPIimg.shape[1], EPIimg.shape[2], EPIimg.shape[3])
filt_EPI = filt_EPI + EPIm[:, :, :, np.newaxis]
filt_EPIimg = nifti.Nifti1Image(filt_EPI, EPIaffine, EPIhdr)
nib.save(filt_EPIimg, args.out)
