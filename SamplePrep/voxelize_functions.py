
import numpy as np
import pandas as pd
from math import factorial
import random
import time
import Bio
import pickle

from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(PERMISSIVE=1)

# Calculate atomic density at a distance from the atom center:
def atom_density(distance,rvw):
    #density = np.exp(-distance**2/(2*std**2))
    if distance == 0:
        return 1 
    else:
        density = 1- np.exp(-(rvw/distance)**12)
        return density
    
# Calculate center of geometry for a given residue:
def res_cog(residue):                                 
    coord = [residue.get_list()[i].get_coord() for i in range(0,np.shape(residue.get_list())[0])]
    cog = np.mean(coord,axis=0)
    return cog

