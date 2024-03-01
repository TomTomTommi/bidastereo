#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

python ./evaluation/evaluate.py --config-name eval_real_data \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.kernel_size=20 \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./logging/finetune3_bdastereo_multi_sf_dr/model_bdastereo_multi_053358.pth