#!/bin/bash

# Make temporary directory for download/unpack
mkdir -p tgz
cd tgz

# Download 
for i in `cat ../ufl_urls.txt`; do echo $i; wget $i; done

# Unpack
for i in `cat ../ufl_matrices.txt`; do gunzip $i.tar.gz; tar -xvf $i.tar; rm $i.tar; done

# Relocate
mkdir -p ../mtx
for i in `find . -name *.mtx`; do echo $i; mv $i ../mtx/; done

# Cleanup
cd ..
rm -rf tgz
