#!/bin/bash
mkdir -p html/mathse
rm -f html/mathse/*

for file in NTCIR11-Math-{1..100}.txt
#for file in NTCIR11-Math-1.txt
do
	query=`python3 query-of.py $file`
	query=`python3 query-specialize.py "$query"`
	echo "$file: $query"
	curl --data-urlencode "q=$query" http://127.0.0.1/cgi/search.cgi > html/mathse/${file}.html
done
