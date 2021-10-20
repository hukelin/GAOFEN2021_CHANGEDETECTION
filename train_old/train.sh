# exp_name="TrainLEVIR_CD_v4"
# data_root="/home/kelin/data/LEVIR-CD"
# batch_size=4
# val_batch_size=4
# epochs=50
# lr=0.0001
# weight_decay=0.01
# backbone="hrnet_w30"
# model="ChangeModel"
# pretrain_from="exp_result/ChangeModel_hrnet_w30_TrainLEVIR_CD_v2_2021-10-11-13:34/checkpoints/ChangeModel_hrnet_w30_bin_44.94.pth"
# # --tta \
# python train_bin_v5.py \
#     --exp_name $exp_name \
#     --data_root $data_root \
#     --batch_size $batch_size \
#     --val_batch_size $val_batch_size \
#     --epochs $epochs \
#     --lr $lr \
#     --weight_decay $weight_decay \
#     --backbone $backbone \
#     --model $model \
#     --pretrain_from $pretrain_from

# #!/usr/bin/env bash
# # general settings
data_root="/home/kelin/data/LEVIR-CD" # 这里改成存放训练集和测试集的文件目录
batch_size=4
val_batch_size=4
epochs=50
lr=0.0001
weight_decay=0.01

# train1-ChangeModel+hrnet_w30
exp_name="change_v1"
model="ChangeModel"
backbone="hrnet_w30"

python train_v5.py \
    --exp_name $exp_name \
    --data_root $data_root \
    --batch_size $batch_size \
    --val_batch_size $val_batch_size \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay \
    --backbone $backbone \
    --model $model