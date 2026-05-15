#!/usr/bin/env bash
set -euo pipefail

echo "Repository status"
git status --short --branch

echo
echo "Project files"
find . -maxdepth 2 -type f | sort

