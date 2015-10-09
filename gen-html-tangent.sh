#!/bin/bash
mkdir -p html/tangent
rm -f html/tangent/*
for rpath in `ls texlist/*.qry`
do
	apath="`pwd`/$rpath"
	file=`basename $rpath`
	curl http://127.0.0.1/cgi/show-tex-list.cgi?p=$apath > html/tangent/${file}.html
done
