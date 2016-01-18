printf '### Running analysis for Figure 6...\n'

# Remove previous files
rm library.txt bin_*.txt files.txt data.txt \
	crp_model.txt

# Make symbolic link to data files
ln -s ../../data/sortseq/full-wt/library.txt library.txt
ln -s ../../data/sortseq/full-wt/files.txt files.txt
for i in `seq 1 9`;
do
	ln -s ../../data/sortseq/full-wt/bin_$i.txt bin_$i.txt
done 

# Create dataset file
printf '### Running sortseq gatherseqs...\n'
sst preprocess -i files.txt -o data.txt

# Run script in figure
printf '### Running sortseq learn_model...\n'
sst learn_matrix --start 3 --end 25 -i data.txt -o crp_model.txt 

# Display snippet
printf '### From data.txt:\n'
head -n 1 data.txt | column -t
printf '\n'

printf '### From crp_matrix.txt:\n'
head -n 1 crp_model.txt | column -t
printf '\n'

printf '### Done!\n'