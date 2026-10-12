#!/bin/zsh
# Rebuild docs/aircraft-in-flight-review.pptx from the result files. Run from the repo root:
#   docs/deck/build.sh
# Needs: .venv (repo requirements), node with pptxgenjs (npm install in docs/deck), the pptx skill's
# apply_theme.js (PPTX_SKILL), and the results under data/ and models/ (see docs/report.md): part 2 and
# clip results, data/s2_flying chips and S2Net results, models/*.pt.
set -e
DECK=docs/deck; B=$DECK/build; PY=.venv/bin/python
mkdir -p $B/img
$PY -m satellite.evaluate_s2 models/satellite_planenet.pt data/s2_airborne --json $B/s2_airborne_eval.json > /dev/null
PYTHONPATH=. $PY $DECK/collect_data.py $B/deck_data.json   # also scores S2Net and PlaneNet on data/s2_airborne
[[ -f $B/pr_curves.json ]] || PYTHONPATH=. $PY $DECK/pr_curves.py $B/pr_curves.json   # about 15 min on CPU
$PY $DECK/s2_figs.py $B/img
$PY $DECK/before_after.py $B
NODE_PATH=$DECK/node_modules node $DECK/build.js "$PWD" $B/deck.pptx
$PY $DECK/postprocess.py $B/deck.pptx docs/aircraft-in-flight-review.pptx
echo "wrote docs/aircraft-in-flight-review.pptx"
