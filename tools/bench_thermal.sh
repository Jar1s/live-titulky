#!/bin/zsh
# Termálny bench: opakuj continuous.wav ~30 min, loguj final decode trend.
# Throttling sa prejaví rastom decode_s medzi iteráciami.
cd "$(dirname "$0")/.." || exit 1
LOG=/tmp/thermal_bench.log
: > $LOG
echo "start: $(date)" >> $LOG
for i in $(seq 1 12); do
  T0=$(date +%s)
  HF_HOME=/Volumes/GRINDCASTER/AI-MODELS/hf-cache \
  TITULKY_SUITE_ONLY=continuous \
  .venv/bin/python test_suite.py 2>/dev/null \
    | grep -E "continuous" | tail -1 \
    | awk -v i=$i -v t=$(( $(date +%s) - T0 )) '{print "iter", i, "(", t "s ):", $0}' >> $LOG
  pmset -g therm 2>/dev/null | grep -E "CPU_Speed_Limit" >> $LOG
done
echo "koniec: $(date)" >> $LOG
