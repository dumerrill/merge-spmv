#!/bin/bash

if [ "$#" -eq 0 ]; then
    MTX_DIR = mtx
else
	MTX_DIR = $1
fi

# Make temporary directory for download/unpack
mkdir -p tgz
cd tgz

# Download 
for i in `cat ../ufl_urls.txt`; do echo $i; wget $i; done

# Unpack
for i in `cat ../ufl_matrices.txt`; do gunzip $i.tar.gz; tar -xvf $i.tar; rm $i.tar; done

# Relocate
mkdir -p ../$MTX_DIR
for i in `find . -name *.mtx`; do echo $i; mv $i ../$MTX_DIR/; done

# Cleanup
cd ..
rm -rf tgz
