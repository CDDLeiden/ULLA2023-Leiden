#!/bin/bash

set -e

ssh-keyscan -H github.com >> ~/.ssh/known_hosts
git clone git@github.com:CDDLeiden/ULLA2023-Leiden.git
rm -rf ULLA2023-Leiden/assignments
