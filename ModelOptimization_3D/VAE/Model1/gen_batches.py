#!/usr/bin/env python3
# -*- coding: utf-8 -*

import numpy as np
import os
import pickle

# Generate batches for training data:
def gen_batches(batch_size):
    while True:

        # Setup list of size batch_size containing paths to
        # patches:

        patch_path = os.getcwd()+'/../../patches/train'
        all_patches = os.listdir(patch_path)  
        batch_paths = np.random.choice(all_patches,size=batch_size)
        label_path = os.getcwd()+'/../../sap_labels.txt'
        batch_input=[]
        batch_label=[]

        for input_path in batch_paths:
            print(patch_path+"/"+input_path)
            pickle_in = open(patch_path+"/"+input_path,"rb")
            input = pickle.load(pickle_in)
          
            # Swap axes to create channels last format
            # (adjust in prep script later)
            input = np.swapaxes(input,0,1)
            input = np.swapaxes(input,1,2)
            input = np.swapaxes(input,2,3)
            #input[input>0.5]=1
            #input[input<0.5]=0
            
            batch_input += [ input ]
            
        batch_input = np.array(batch_input)
        
        
        yield(batch_input, batch_input)
        
        
# Generate batches for validation data:
        
def gen_batches_validation(batch_size):
    while True:

        # Setup list of size batch_size containing paths to
        # patches:

        patch_path = os.getcwd()+'/../../patches/validation'
        all_patches = os.listdir(patch_path)  
        batch_paths = np.random.choice(all_patches,size=batch_size)
        label_path = os.getcwd()+'/../../sap_labels.txt'
        batch_input=[]
        batch_label=[]

        for input_path in batch_paths:
            pickle_in = open(patch_path+"/"+input_path,"rb")
            input = pickle.load(pickle_in)

            # Swap axes to create channels last format
            # (adjust in prep script later)
            
            input = np.swapaxes(input,0,1)
            input = np.swapaxes(input,1,2)
            input = np.swapaxes(input,2,3)
            #input[input>0.5]=1
            #input[input<0.5]=0
            
            batch_input += [ input ]
            
        batch_input = np.array(batch_input)
        
        
        yield(batch_input, batch_input)

# Generate batches for test data:
def gen_batches_test(batch_size):
    while True:

        # Setup list of size batch_size containing paths to
        # patches:

        patch_path = os.getcwd()+'/../../patches/test'
        all_patches = os.listdir(patch_path)  
        batch_paths = np.random.choice(all_patches,size=batch_size)
        label_path = os.getcwd()+'/../../sap_labels.txt'
        batch_input=[]
        batch_label=[]

        for input_path in batch_paths:
            print(patch_path+"/"+input_path)
            pickle_in = open(patch_path+"/"+input_path,"rb")
            input = pickle.load(pickle_in)
            
            # Swap axes to create channels last format
            # (adjust in prep script later)
            input = np.swapaxes(input,0,1)
            input = np.swapaxes(input,1,2)
            input = np.swapaxes(input,2,3)
            #input[input>0.5]=1
            #input[input<0.5]=0
            
            batch_input += [ input ]
            
        batch_input = np.array(batch_input)
      
        
        yield(batch_input, batch_input)
        
