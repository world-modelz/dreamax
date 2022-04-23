#!/bin/bash
cp ../requirements.txt .
docker build -t dreamax:v3 .
rm ./requirements.txt
