#!/bin/bash
# Rebuild the root README from its parts. Usage: assemble_readme.sh TOP.md RESULTS.md BOTTOM.md
set -euo pipefail
cat "$1" "$2" "$3" > README.md
