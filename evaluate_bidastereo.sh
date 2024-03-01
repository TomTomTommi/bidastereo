#!/bin/bash

export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH

# evaluate on [sintel, dynamicreplica]

python ./evaluation/evaluate.py --config-name eval_bidastereo_sintel_clean \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.kernel_size=50 \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr.pth

python ./evaluation/evaluate.py --config-name eval_bidastereo_sintel_final \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.kernel_size=50 \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr.pth

python ./evaluation/evaluate.py --config-name eval_bidastereo_dynamic_replica \
MODEL.model_name=BiDAStereoModel \
MODEL.BiDAStereoModel.type=bidastereo \
MODEL.BiDAStereoModel.model_weights=./checkpoints/bidastereo_sf_dr.pth

