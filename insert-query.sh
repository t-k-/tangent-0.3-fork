#!/bin/bash
for path in `ls texlist/*.txt`
do
	file=`basename $path`
	query=`python3 query-of.py $file`
	echo "$file: $query"
	sed -e "1i=$query" $path > ${path}.qry
done
