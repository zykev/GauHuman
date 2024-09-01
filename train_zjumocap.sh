CUDA_VISIBLE_DEVICES=0 python train.py \
-s "/home/jli/datasets/ZJUMoCap/zju_mocap/my_377" \
--exp_name zju_mocap_100_smpl_reduce \
--motion_offset_flag \
--smpl_type smpl \
--actor_gender neutral \
--iterations 1200 