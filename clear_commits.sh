#!/bin/bash

git checkout --orphan new-branch

git add -A
git commit -m "Initial commit"

git branch -D main

git branch -m new-branch main

git push -f origin main