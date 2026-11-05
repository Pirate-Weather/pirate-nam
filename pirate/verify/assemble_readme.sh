#!/bin/bash
# Rebuild the Pirate NAM part of the root README from its parts, keeping upstream's README below the marker.
# Usage: assemble_readme.sh TOP.md RESULTS.md BOTTOM.md
set -euo pipefail
MARK='*Everything below is WOOF-NAM'"'"'s own README, unchanged.*'
tail -n +$(( $(grep -nF "$MARK" README.md | cut -d: -f1) + 2 )) README.md > /tmp/upstream_readme.$$
cat "$1" "$2" "$3" /tmp/upstream_readme.$$ > README.md; rm /tmp/upstream_readme.$$
