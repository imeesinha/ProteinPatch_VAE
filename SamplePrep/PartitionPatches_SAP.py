import numpy as np
import pandas as pd
import random
import time
import Bio
import pickle
import os

import voxelize_functions
from voxelize_functions import *
import atom_types
from atom_types import atom_id
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio import PDB
from Bio.PDB import *

parser = PDBParser(PERMISSIVE=1)
io = PDB.PDBIO()
start = time.time()

# Settings:
cutoff = 1.5 # cutoff in one direction in angstroms
rvw = [1.7,1.55,1.52,1.8] # Van der Waal radii in angstroms
dx = 0.3 # 0.3 angstrom grid size
buffer = 20 # Buffer size around edges of protein in angstroms
n_atomtypes = 4 # Number of types of atoms, must be aligned with atom_types.py
side_length = 14.4 # 14.4 Angstroms patch length
full_length = 48 # 48 x 48 x 48 grid
exposure_cutoff = 0.20 # Fractional SASA exposure cutoff


Patches = np.zeros([1,n_atomtypes,full_length,full_length,full_length])


#references from Creighton et al. 1993
sasa_dict = {'ALA':1.13, 'ARG':2.41, 'ASN':1.58, 'ASP':1.51, 'ASH': 1.51, 'CYS':1.40, 'CYX':1.40, 'GLN':1.89, 'GLU':1.83, 'GLH':1.83, 
             'GLY':0.85, 'HIS':1.94, 'HIE':1.94, 'HID':1.94, 'HIP':1.94, 'ILE':1.82, 'LEU':1.80, 'LYS':2.11, 'LYN':2.11,
             'MET':2.04, 'PHE':2.18, 'PRO':1.43, 'SER':1.22, 'THR':1.46, 'TRP':2.59, 'TYR':2.29, 'VAL':1.60}


# Identify file locations:
curr_path = os.getcwd()
# Pickle files are in a subdirectory named "4pickle_perpdb"
pdb_path = os.path.join(curr_path,'3clean_sap_pdb') 
pickle_path = os.path.join(curr_path,'4pickle_combined')
combined_path = os.path.join(curr_path,'5pickle_combined')
patch_path = os.path.join(curr_path,'6patches')
labels_path = os.path.join(curr_path, '7labels')
all_pickles = os.listdir(pickle_path)

k=1
start_proc=time.time()
patchnum=0
label_sheet = open("sap_labels.txt","w") 
for item in all_pickles: 
    file, extension = os.path.splitext(item)
    if (extension == '.pickle'):
        
        # For every structure load in occupancy grid, SAP grid and structure:
        print(item, k)
        pickle_in = open(pickle_path+'/'+item,"rb")
        atomic_occupancy = pickle.load(pickle_in)
        pickle_in = open(combined_path+'/'+file+"-sap.pickle","rb")
        sap_occupancy = pickle.load(pickle_in)
        structure = parser.get_structure(file, pdb_path+'/'+file+'-clean-sap.pdb')
   
        k+=1
        
        # Reading sasa-xvg file:
        resarea=[]
        with open(pdb_path+'/'+file+'-resarea.xvg') as f:
            for line in f:
                cols = line.split()
                if str(cols[0][0]) != '#' and str(cols[0][0]) != '@':
                    resarea.append(float(cols[1]))
        
        # Define grid edges from structure:
        coord_list = [atom.coord for atom in structure.get_atoms()]
        
        xmin = min([coord_list[i][0] for i in range(0,np.shape(coord_list)[0])])
        xmin = xmin-buffer
        xmax = max([coord_list[i][0] for i in range(0,np.shape(coord_list)[0])])
        xmax = xmax+buffer
        
        ymin = min([coord_list[i][1] for i in range(0,np.shape(coord_list)[0])])
        ymin = ymin-buffer
        ymax = max([coord_list[i][1] for i in range(0,np.shape(coord_list)[0])])
        ymax = ymax+ buffer
        
        zmin = min([coord_list[i][2] for i in range(0,np.shape(coord_list)[0])])
        zmin = zmin-buffer
        zmax = max([coord_list[i][2] for i in range(0,np.shape(coord_list)[0])])
        zmax = zmax+buffer

        linx = np.arange(xmin,xmax,dx)
        liny = np.arange(ymin,ymax,dx)
        linz = np.arange(zmin,zmax,dx)        

        j=-1
        for residue in list(structure.get_residues()):
            j+=1
            rasa = resarea[j]/sasa_dict[residue.get_resname()]
            #print(rasa)
            if float(rasa) > exposure_cutoff:
                #print ('SASA cutoff verified')
                # Calculate residue center:
                center = res_cog(residue)
                # Calculate start/end positions:
                vox_start = center-side_length/2.0
                vox_end = center+side_length/2.0
                    
                x_start=np.where((linx>vox_start[0])&(linx<vox_end[0]))[0][0]
                x_end=np.where((linx>vox_start[0])&(linx<vox_end[0]))[0][-1]
                    
                y_start=np.where((liny>vox_start[1])&(liny<vox_end[1]))[0][0]
                y_end=np.where((liny>vox_start[1])&(liny<vox_end[1]))[0][-1]
                    
                z_start=np.where((linz>vox_start[2])&(linz<vox_end[2]))[0][0]
                z_end=np.where((linz>vox_start[2])&(linz<vox_end[2]))[0][-1]
                    
                patch = atomic_occupancy[:,x_start:(x_end+1),y_start:(y_end+1),z_start:(z_end+1)]
                sap = sap_occupancy[:,x_start:(x_end+1),y_start:(y_end+1),z_start:(z_end+1)]
                #print (patch.shape)
                    
                if np.shape(patch)==(n_atomtypes, 48, 48, 48):
                    #print('Shape verified')
                    resname = residue.get_resname()
                    count = np.count_nonzero(sap)
                    SAPValue = np.true_divide(np.sum(sap),count)
                    label_sheet.write('patch'+ str(patchnum)+'.pickle   '+ str(file)+ '_' + str(resname) + str(j+1)+'   '+str(SAPValue)) 
                    label_sheet.write("\n") 
                    pname='patch'+ str(patchnum)+'.pickle'
                    patchnum+=1
                    picklename = os.path.join(patch_path,pname) 
                    pickle_out = open(picklename,"wb")
                    pickle.dump(patch,pickle_out)
                    pickle_out.close()
                        
                        
end = time.time()
label_sheet.close()
print('Time Elapsed in minutes:',(end-start)/60)
           
