#!/usr/bin/env bash

# bash tools/dist_train.sh configs/0cd_ce_baseline_me/changeclip_whucd.py 4 --work-dir work_dirs_baseline_me/changeclip_whucd
# bash tools/test.sh WHUB configs/0seg/baseline_whub.py 4 work_dirs/baseline_whub

# bash tools/test.sh WHUB configs/0seg/mmseg_text_whub.py 4 work_dirs/mmseg_text_whub

# bash tools/dist_train.sh configs/0seg/buildformer.py 2 --work-dir work_dirs/buildformer
# bash tools/test.sh WHUB configs/0seg/buildformer.py 2 work_dirs/buildformer

# bash tools/dist_train.sh configs/0seg/baseline_whub.py 2 --work-dir work_dirs/baseline_whub
# bash tools/test.sh WHUB configs/0seg/baseline_whub.py 2 work_dirs/baseline_whub

# bash tools/dist_train.sh configs/0seg/mmseg_whub.py 2 --work-dir work_dirs/mmseg_whub
# bash tools/test.sh WHUB configs/0seg/mmseg_whub.py 2 work_dirs/mmseg_whub

# bash tools/dist_train.sh configs/0seg/mmseg_text_whub.py 2 --work-dir work_dirs/mmseg_text_whub
# bash tools/test.sh WHUB configs/0seg/mmseg_text_whub.py 2 work_dirs/mmseg_text_whub

# bash tools/dist_train.sh configs/0seg/mmseg_text_swin_whub.py 2 --work-dir work_dirs/mmseg_text_swin_whub
# bash tools/test.sh WHUB configs/0seg/mmseg_text_swin_whub.py 2 work_dirs/mmseg_text_swin_whub
