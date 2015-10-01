#!/bin/bash
output_dir='texlist'
mkdir -p "$output_dir"
output="${output_dir}/texlist.txt"

process_query_name() {
	query_name=`echo "$1" | awk '{print $2}'`
	output="${output_dir}/${query_name}.txt"
	> "$output"
}

process_result() {
	docID=`echo "$1" | awk '{print $2}'`
	pos=`echo "$1" | awk '{print $3}'`
	python3 doc.py $docID $pos | grep '<math' | grep -Po '(?<=alttext=").*?(?=")' | tee -ai "$2"
}

cnt=0
while read line
do
	echo "$line" | awk '{if($1 == "Q") exit 0; else exit 1;}' && \
		process_query_name "$line"

	echo "$line" | awk '{if($1 == "R") exit 0; else exit 1;}' && \
		let 'cnt=cnt+1' && echo -n "${cnt}: " && \
		process_result "$line" "$output"

	if [ $cnt -gt 30 ]; then
		echo '[break]'
		break
	fi
done < reranked_results.tsv
