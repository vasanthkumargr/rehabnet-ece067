# Milestone 1 on Kaggle: data, GPU check, pose extraction, validation, invariance test.
pip install -q mediapipe dtaidistance 2>&1 | tail -1
python -c "import mediapipe; print('mediapipe', mediapipe.__version__)"
bash scripts/kaggle_fetch_rehab24.sh
# --- speed benchmark + decide CPU or GPU delegate
python -P scripts/m1_speed_bench.py kaggle_t4 2>&1 | grep -v "W0000\|I0000\|INFO" | tee logs_speed.txt || true
DEL=$(grep BEST_DELEGATE logs_speed.txt | cut -d= -f2); export REHABNET_DELEGATE=${DEL:-cpu}
echo "using delegate $REHABNET_DELEGATE"
W=$NPROC; [ "$REHABNET_DELEGATE" = gpu ] && W=4
# --- all 130 video streams at 10 fps, and 6 recordings (12 streams) at 30 fps
python -P scripts/m1_extract_rehab24.py --every 3 --workers $W 2>&1 | grep "^\[\|Error\|error" | tail -200
python -P scripts/m1_extract_rehab24.py --every 1 --workers $W --only PM_000,PM_003,PM_010,PM_005,PM_021,PM_008 2>&1 | grep "^\[\|Error\|error"
# --- validation vs OptiTrack
VARIANT=full_every3 python -P scripts/m1_validate.py 2>&1 | grep -v "^done" | tail -80
VARIANT=full TAG=_30fps python -P scripts/m1_validate.py 2>&1 | grep -v "^done" | tail -20
# --- distance / position invariance test
python -P scripts/m1_normalization_test.py 2>&1 | grep -v "W0000\|I0000\|INFO\|model_load\|arning"
mkdir -p logs && cp logs_speed.txt logs/
