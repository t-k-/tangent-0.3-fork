#!/bin/sh
rm -f results-db-index.tsv *.cache reranked_results.tsv

python3 query.py tangent.cntl
echo "S" | cat db-index/* - | tangent/mathindex.exe  > results-db-index.tsv
python3 rerank_results.py tangent.cntl results-db-index.tsv 4 reranked_results.tsv
