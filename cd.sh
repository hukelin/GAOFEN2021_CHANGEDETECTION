# !/usr/bin/env bash
# general settings
for fold in 4; do
    train_root="datasets/data/train_${fold}" # 这里改成存放训练集和测试集的文件目录
    val_root="datasets/data/val_${fold}"     # 这里改成存放训练集和测试集的文件目录
    batch_size=4
    val_batch_size=4
    epochs=50
    lr=0.0001
    weight_decay=0.0001
    optimizer="adamw"
    # pretrain_from="exp_result2/ChangeModel_resnext50_32x4d_change_v2_2021-10-14-11:37/checkpoints/ChangeModel_resnext50_32x4d_bin_45.49.pth"

    # train1-ChangeModel+hrnet_w30
    exp_name="fold${fold}"
    model="Siamese"
    backbone="hrnet_w30"

    python train_v2.py \
        --exp_name $exp_name \
        --train_root $train_root \
        --val_root $val_root \
        --batch_size $batch_size \
        --val_batch_size $val_batch_size \
        --epochs $epochs \
        --lr $lr \
        --weight_decay $weight_decay \
        --backbone $backbone \
        --model $model \
        --optimizer $optimizer
done
# # train2-ChangeModel+resnext50_32x4d
# exp_name="change_v2"
# model="ChangeModel"
# backbone="resnext50_32x4d"

# python train_v1.py \
#     --exp_name $exp_name \
#     --data_root $data_root \
#     --batch_size $batch_size \
#     --val_batch_size $val_batch_size \
#     --epochs $epochs \
#     --lr $lr \
#     --weight_decay $weight_decay \
#     --backbone $backbone \
#     --model $model \
#     --optimizer $optimizer
# # --pretrain_from $pretrain_from

# # train3-Siamese+hrnet_w30
# exp_name="change_v3"
# model="Siamese"
# backbone="hrnet_w30"

# python train_v2.py \
#     --exp_name $exp_name \
#     --data_root $data_root \
#     --batch_size $batch_size \
#     --val_batch_size $val_batch_size \
#     --epochs $epochs \
#     --lr $lr \
#     --weight_decay $weight_decay \
#     --backbone $backbone \
#     --model $model \
#     --optimizer $optimizer

# # train4-Siamese+resnext50_32x4d
# exp_name="change_v4"
# model="Siamese"
# backbone="resnext50_32x4d"

# python train_v2.py \
#     --exp_name $exp_name \
#     --data_root $data_root \
#     --batch_size $batch_size \
#     --val_batch_size $val_batch_size \
#     --epochs $epochs \
#     --lr $lr \
#     --weight_decay $weight_decay \
#     --backbone $backbone \
#     --model $model \
#     --optimizer $optimizer

# # test
# # python test.py $data_root
