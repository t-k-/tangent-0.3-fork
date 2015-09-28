#!/bin/sh
find *.tex -exec echo -n {}': ' \; -exec cat {} \;
