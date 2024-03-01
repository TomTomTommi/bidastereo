#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

python train_bidastereo.py --name bidastereo --batch_size 2 \
 --spatial_scale -0.2 0.4 --image_size 256 256 --saturation_range 0 1.4 --num_steps 80000 --finetune_step 39999 \
 --ckpt_path logging/bidastereo_sf_dr  \
 --sample_len 5 --lr 0.0004 --train_iters 10 --valid_iters 20    \
 --num_workers 16 --save_freq 100 --train_datasets dynamic_replica things monkaa driving
