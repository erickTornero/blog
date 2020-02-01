#!/bin/bash

hugo -t indigo

git add .

git commit -m "$1"

git push origin master

