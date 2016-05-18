# Supplemental Information Pipeline

echo "Simulating library..."
sortseq_tools simulate_library -w TAATGTGAGTTAGCTCACTCAT -n 100000 -m 0.24 -o library.txt

echo "Simulating Sort-Seq experiment..."
sortseq_tools simulate_sort -m true_model.txt -n 4 -i library.txt -o dataset.txt

echo "Computing mutation profile..."
sortseq_tools profile_mut -b 0 -i dataset.txt -o mutprofile.txt

echo "Computing frequency profile"
sortseq_tools profile_freq -b 0 -i dataset.txt -o freqprofile.txt

echo "Computing information profile"
sortseq_tools profile_info --err -i dataset.txt -o infoprofile.txt

echo "Learning a matrix model using least squares optimization..."
sortseq_tools learn_model -lm LS -mt MAT -i dataset.txt -o matrix_model.txt

echo "Learning a neighbor model using least squares optimization..."
sortseq_tools learn_model -lm LS -mt NBR -i dataset.txt -o neighbor_model.txt

echo "Evaluating model on dataset..."
sortseq_tools evaluate_model -m matrix_model.txt -i dataset.txt -o dataset_with_values.txt

echo "Scanning model over genome..."
sortseq_tools scan_model -n 100 -m matrix_model.txt -i genome_ecoli.fa -o genome_sites.txt

echo "Computing predictive information for inferred model..."
sortseq_tools predictiveinfo -m matrix_model.txt -ds dataset.txt

echo "Computing predictive information for true model..."
sortseq_tools predictiveinfo -m true_model.txt -ds dataset.txt
