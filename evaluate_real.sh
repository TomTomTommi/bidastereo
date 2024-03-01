#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

python ./evaluation/evaluate.py --config-name eval_real_data \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.kernel_size=20 \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr.pth