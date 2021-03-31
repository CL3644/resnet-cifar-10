#!/bin/bash

for model in resnet56
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model |& tee -a log_$model"
    python -u trainer.py  --arch=$model --weight-decay=0 --save-dir=save_$model |& tee -a log_$model
done