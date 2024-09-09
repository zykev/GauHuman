expname=zju_mocap/my_377_100_smpl_featuregs_wocolor

CUDA_VISIBLE_DEVICES=0 python render.py \
-m output/${expname} \
--motion_offset_flag \
--smpl_type smpl \
--actor_gender neutral \
--iteration 2000 \
--skip_train