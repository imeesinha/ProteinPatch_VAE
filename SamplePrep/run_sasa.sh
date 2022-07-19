#! /bin/bash
source ~/.bashrc
source /usr/local/gromacs/bin/GMXRC
# Replace the following with your own environment location
source activate /home/sinhai/anaconda3/envs/tf

# Remove waters
# Fix hydrogens
# Remove heteroatoms

curr=`pwd`
cd $curr/1raw_pdb
for i in *.pdb
do
    pdb="$i"
    fname=${pdb%.*}
    pdb4amber -i $i -o "temp0.pdb" --dry 
    sed -i '/HETATM/d' temp0.pdb
    echo 0 |gmx sasa -f temp0.pdb -s temp0.pdb -or $fname"-resarea.xvg" 
    sed 's/ CYX / CYS /' temp0.pdb > temp1.pdb
    sed 's/ CYM / CYS /' temp1.pdb > temp2.pdb
    sed 's/ HID / HIS /' temp2.pdb > temp3.pdb
    sed 's/ HIE / HIS /' temp3.pdb > temp4.pdb
    sed 's/ HIP / HIS /' temp4.pdb > clean.pdb
    mv clean.pdb ../2clean_pdb/$fname"-clean.pdb"
    
    mv $fname"-resarea.xvg" ../3clean_sap_pdb/
    rm temp*
        
done
cd ..

