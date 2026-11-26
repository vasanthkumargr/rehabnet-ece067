# Fetch REHAB24-6 (30 fps joints + videos) and the MediaPipe pose models inside a Kaggle kernel.
B="https://zenodo.org/records/13305826/files"
mkdir -p data/rehab24_6 models
curl -sSL -o data/rehab24_6/Segmentation.csv "$B/Segmentation.csv?download=1"
python -u scripts/remote_zip_extract.py "$B/3d_joints.zip?download=1" 551298977 data/rehab24_6/joints3d 30fps 16 | tail -1
python -u scripts/remote_zip_extract.py "$B/2d_joints.zip?download=1" 693401748 data/rehab24_6/joints2d 30fps 16 | tail -1
python -u scripts/remote_zip_extract.py "$B/videos.zip?download=1" 2651613914 data/rehab24_6/videos .mp4 16 | tail -1
for v in lite full heavy; do
  curl -sSL -o models/pose_landmarker_$v.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_$v/float16/latest/pose_landmarker_$v.task
done
ls data/rehab24_6/videos/*/*.mp4 | wc -l
