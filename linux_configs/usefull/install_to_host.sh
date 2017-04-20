#!/bin/bash
scp -r $(ls -a | grep -v -E '\.\.' | grep -v -E '^\.$') $1
