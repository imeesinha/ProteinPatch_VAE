#! /bin/bash
source ~/.bashrc

curr=`pwd`
codes=$curr/common-sap
clean=$curr/../2clean_pdb
clean_sap=$curr/../3clean_sap_pdb

cp $clean/*.pdb .
for i in *.pdb
do
    pdb="$i"
    fname=${pdb%.*}
    mkdir $fname
    mv $i $fname/
    cd $fname/
    cp -rf $codes/* .
    ./sap -pdb $fname".pdb" 
    mv outsap.pdb $fname"-sap.pdb"
    cp $fname"-sap.pdb" $clean_sap/
    cd ..
done

    
    
    
    
    
     




