# Milestone 5 on Kaggle: robustness / model-size / speed test of the camera pipeline.

pip install -q mediapipe dtaidistance 2>&1 | tail -1
bash scripts/kaggle_fetch_rehab24.sh
mkdir -p data/processed results/m5
ls -la data/processed/
python -P scripts/m5_robustness.py 2>&1 | grep -v "W0000\|I0000\|INFO\|model_load\|arning" | tail -60
