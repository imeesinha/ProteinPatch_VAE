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


# Initialize:
proc_file = 0 # Number of files processed

# Identify file locations:
curr_path = os.getcwd()
# PDB files are in a subdirectory named "3clean_sap_pdb"
pdb_path = os.path.join(curr_path,'3clean_sap_pdb') 
pickle_path = os.path.join(curr_path,'4pickle_combined')
combined_path = os.path.join(curr_path,'5pickle_combined')
all_files = os.listdir(pdb_path)

# Load in PDB files:

for item in all_files:
    file, extension = os.path.splitext(item)
    if ((extension == '.pdb')&(file[-10:]=='-clean-sap')):
        proc_file +=1
        
        print('Processing File', proc_file, file)
        
        structure_id = file
        filename = os.path.join(pdb_path,item)
        structure = parser.get_structure(structure_id,filename)
       
        # Populate a grid with atomic densities:
        
        # Define grid edges
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

        gridx, gridy, gridz = np.meshgrid(linx,liny,linz)
        gridshape = np.shape(gridx)
        
        # Fill densities into grid
        
        atomic_occupancy = np.zeros([n_atomtypes,np.shape(linx)[0],np.shape(liny)[0],np.shape(linz)[0]])
        sap_occupancy = np.zeros([1,np.shape(linx)[0],np.shape(liny)[0],np.shape(linz)[0]])
        atomcount = 0
        
        for atom in structure.get_atoms():
            lowest = 5
            id_mat = atom_id(atom)
            atomcount = atomcount+1
            for i in range(0,n_atomtypes):
                if id_mat[i]==1:
                    atomcoord = atom.get_coord()
                    atomsap = atom.get_bfactor() 
                    
                    for x in np.where(abs(linx-atomcoord[0])<= cutoff)[0]:
                        for y in np.where(abs(liny-atomcoord[1])<= cutoff)[0]:
                            for z in np.where(abs(linz-atomcoord[2])<= cutoff)[0]:
                                pointcoord = np.array([linx[x],liny[y],linz[z]])
                                distance = np.linalg.norm(pointcoord-atomcoord)
                                atomic_occupancy[i,x,y,z] += atom_density(distance,rvw[i])
                                if distance < lowest:
                                    xlow, ylow, zlow = x, y, z
                                    lowest = distance
            sap_occupancy[0,xlow,ylow,zlow] = atomsap           
                                
        print ('Total no. of atoms processed', atomcount)                        
        pname = structure_id[:4] + '.pickle'
        picklename = os.path.join(pickle_path,pname)    
        pickle_out = open(picklename,"wb")
        pickle.dump(atomic_occupancy, pickle_out)
        pickle_out.close()
        
        pname = structure_id[:4] + '-sap.pickle'
        picklename = os.path.join(combined_path,pname)    
        pickle_out = open(picklename,"wb")
        pickle.dump(sap_occupancy, pickle_out)
        pickle_out.close()

#print('Unprocessed files: ',(len(all_files)/2-proc_file))
end = time.time()
print('Time Elapsed in minutes: ', np.round((end-start)/60))
