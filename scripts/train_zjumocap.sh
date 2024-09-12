expname=zju_mocap/my_377_100_smpl_featuregs_color3
mkdir -p output/${expname}

CUDA_VISIBLE_DEVICES=3 python train.py \
-s /home/jli/datasets/ZJUMoCap/zju_mocap/my_377 \
--eval \
--exp_name ${expname} \
--motion_offset_flag \
--smpl_type smpl \
--actor_gender neutral \
--iterations 2000 \
--semantic_feature_dim 64 | tee output/${expname}/train.log