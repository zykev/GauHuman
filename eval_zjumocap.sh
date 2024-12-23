CUDA_VISIBLE_DEVICES=0 python render.py \
-m output/zju_mocap_100_smpl_reduce \
--motion_offset_flag \
--smpl_type smpl \
--actor_gender neutral \
--iteration 1200 \
--skip_train