#!/bin/bash
for path in `ls texlist/*.txt`
do
	file=`basename $path`
	query=`python3 query-of.py $file`
	echo "$file: $query"
	echo "\\text{query}: \\qquad $query" > ${path}.qry
	cat $path >> ${path}.qry
done
