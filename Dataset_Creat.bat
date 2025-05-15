@echo off

echo 1. Preprocess CCVID (silhouette + skeleton create)
python preprocess_ccvid.py

echo 2. Preprocess CCVID PKL conversion
python preprocess_ccvid_pkl.py

echo 3. Generate Meta JSON
python generate_meta_json.py

echo Finish
pause
