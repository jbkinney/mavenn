printf '### Cleaning sortseq directory...\n'

printf '### Reformatting raw data...\n'
cd '_raw'
python preprocess.py
cd ..


printf '### Writing files.txt, data.txt, and fitting models to each data set...\n'

EXPERIMENTS='full-wt crp-wt rnap-wt full-0 full-150 full-500'

for EXPERIMENT in $EXPERIMENTS; 
do 
	# Change to directory
	cd $EXPERIMENT
	echo `pwd`':'
	
	printf "bin\tfile\n" > files.txt
	printf "0\tlibrary.txt\n" >> files.txt
	NUMFILES=`ls bin*.txt | wc -w`
	for I in `seq 1 $NUMFILES`;
	do
		printf "$I\tbin_$I.txt\n" >> files.txt
	done

	# Gather data into data.txt
	COMMAND="cat files.txt"
	echo $COMMAND
	$COMMAND

	# Gather data into data.txt
	COMMAND="sortseq gatherseqs -i files.txt -o data.txt"
	echo $COMMAND
	$COMMAND

	# Fit a model
	if [ "$EXPERIMENT" != "rnap-wt" ] && [ "$EXPERIMENT" != "full-0" ];
	then
		echo "### Fit a crp model..."
		COMMAND="sortseq learn_matrix --start 3 --end 25 -i data.txt -o crp_model.txt"
		echo $COMMAND
		$COMMAND
	fi

	if [ "$EXPERIMENT" != "crp-wt" ];
	then
		echo "### Fit an rnap model..."
		COMMAND="sortseq learn_matrix --start 36 --end 75 -i data.txt -o rnap_model.txt"
		echo $COMMAND
		$COMMAND
	fi

	# Get out of directory
	echo ''
	cd ..
done

printf '### Done!\n'
