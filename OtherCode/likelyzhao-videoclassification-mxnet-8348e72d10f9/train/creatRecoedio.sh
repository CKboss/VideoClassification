SIZE=320
python tools/im2rec.py --list True --recursive True --train-ratio 0.95 ucf-frame /workspace/data/frame/
python tools/im2rec.py --resize $SIZE --quality 90 --num-thread 16  ucf-frame_train /workspace/data/frame/
python tools/im2rec.py --resize $SIZE --quality 90 --num-thread 16  ucf-frame_val /workspace/data/frame/
