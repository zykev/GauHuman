CUDA_VISIBLE_DEVICES=0 python train.py \
-m output/zju_mocap_100_smpl_reduce \
--motion_offset_flag \
--smpl_type smpl \
--actor_gender neutral \
--iterations 1200 \
--skip_train \
--video