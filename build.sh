#!/bin/bash

cd build
apptainer build --fakeroot retnet.sif retnet.def
