#!/bin/bash

set -e

mkdir -p .ssh
ssh-keyscan -H github.com >> ~/.ssh/known_hosts
git clone https://github.com/CDDLeiden/ULLA2023-Leiden.git
rm -rf ULLA2023-Leiden/assignments
