echo "seq" > ../sites.txt
cat MA0049.1.sites | grep -v ">" >> ../sites.txt
cat ../sites.txt