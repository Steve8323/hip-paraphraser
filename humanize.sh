#!/usr/bin/env bash
# One-pass, detector-guided humanizer. Prints ONLY the humanized text.
# A single round: sample candidates and keep drawing (up to --max-k) until the
# local desklib proxy rates the output >= target P(human).
#   echo "Your AI-sounding paragraph." | ./humanize.sh
#   ./humanize.sh --file draft.txt
#   ./humanize.sh --target 0.95 --max-k 60 --file draft.txt
set -euo pipefail
cd "$(dirname "$0")"
exec python3 humanize_best.py --quiet "$@"
