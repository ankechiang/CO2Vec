#!/bin/bash
data='~/CO2Vec/datasets/'
weight='~/CO2Vec/weights/'
trial='0' #run many trials to derived unbiased results
cons='8000' 
pretrain=''
epochs=101
batch_size=512
folder='univ'

for strategy in 'univ' #positive and negative pairs selection
do
	for dim in 16 #dimensionality
	do
		for fold in '0' '1' '2' '3' '4' #5-fold
		do		
			echo "${data}${folder}/t_${trial}c_${cons}f_${fold}/${strategy}/"
			
			mv ${data}${folder}'/t_'${trial}'c_'${cons}'f_'${fold}'/'${strategy}'/' ${data}
			python main.py -d ${strategy} -h1 ${dim} -bs ${batch_size} -e ${epochs} -gt '' -m WVOE
			mv ${data}${strategy}'/' ${data}${folder}'/t_'${trial}'c_'${cons}'f_'${fold}'/'
			
			ndir=${weight}${folder}'_t_'${trial}'c_'${cons}'f_'${fold}'/'
			if [ !  -d "$ndir" ]; then mkdir $ndir
			fi 

			ndir=${weight}${folder}'_t_'${trial}'c_'${cons}'f_'${fold}'/'${strategy}'/'
			if [ !  -d "$ndir" ]; then mkdir $ndir
			fi

			fileLoss=${weight}${strategy}'lstTrainingLoss.'${dim}${pretrain}'.npy'
			if [ -f "$fileLoss" ]; then mv ${fileLoss} ${ndir}
			fi
			fileLoss=${weight}${strategy}'lstLossOc.'${dim}${pretrain}'.npy'
			if [ -f "$fileLoss" ]; then mv ${fileLoss} ${ndir}
			fi
			fileLoss=${weight}${strategy}'lstLossOs.'${dim}${pretrain}'.npy'
			if [ -f "$fileLoss" ]; then mv ${fileLoss} ${ndir}
			fi
			fileLoss=${weight}${strategy}'lstLossLcs.'${dim}${pretrain}'.npy'
			if [ -f "$fileLoss" ]; then mv ${fileLoss} ${ndir}
			fi

			fileWeight=${weight}${strategy}'uid_userDoc.weight.data.'${dim}${pretrain}'.npy'
			if [ -f "$fileWeight" ]; then mv ${fileWeight} ${ndir}
			fi	
		done							
	done
done