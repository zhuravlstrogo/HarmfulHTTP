#!/usr/bin/env bash

docker run \
	--cpus=16 -- memory=300g \ 
	make-cluster:1.0.0