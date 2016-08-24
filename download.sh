#!/bin/bash

mkdir -p tgz

cd tgz

for i in `cat ../ufl_urls.txt`; do echo $i; wget $i; done
