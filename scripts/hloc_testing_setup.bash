# run hloc docker image

DATA_FOLDER="/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/"
CODE_FOLDER="/Users/rdcrlrka/Research/SkySat-Stereo/skysat_stereo_snow"

docker run --rm -it \
--memory=2g \
--publish 8888:8888 \
--volume $DATA_FOLDER:/app/data \
--volume $CODE_FOLDER:/app/code \
hloc:latest
