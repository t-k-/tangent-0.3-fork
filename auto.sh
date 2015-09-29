#!/bin/sh
rm -f results-db-index.tsv *.cache reranked_results.tsv
rm -rf db-index

# index
python3 index.py tangent.cntl

# query
python3 query.py tangent.cntl

# search
echo "S" | cat db-index/* - | tangent/mathindex.exe  > results-db-index.tsv

# rerank
python3 rerank_results.py tangent.cntl results-db-index.tsv 4 reranked_results.tsv

# echo "\\sqrt{b^2-4ac}" | latexmlmath --preload=amsmath --preload=amsfonts --preload=/home/tk/Downloads/Tangent_Code/tangent/math/mws.sty.ltxml - 
