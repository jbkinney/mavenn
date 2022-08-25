#python3 Titeseq_search_script.py 5.0
#python3 Titeseq_search_script.py 6.0


echo "sigma 1.0"
for i in {1..3}
do
    for j in {4..12}
    do 
    	echo "mu_pos $j mu_neg $i"
        python3 Titeseq_search_script.py $j $i 1.0
    done
done

echo "sigma 2.0"
for i in {1..3}
do
    for j in {4..12}
    do 
	    echo "mu_pos $j mu_neg $i"
        python3 Titeseq_search_script.py $j $i 2.0
    done
done