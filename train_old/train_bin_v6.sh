exp_name="change_Norm_v1"
data_root="/home/kelin/data/LEVIR-CD"
batch_size=4
val_batch_size=4
epochs=50
lr=0.0001
weight_decay=0.0001
backbone="hrnet_w30"
model="farseg"
# pretrain_from="exp_result/farseg_hrnet_w30_change_Norm_v1_2021-10-09-23:01/checkpoints/farseg_hrnet_w30_bin_64.15.pth"
# pretrained=false
# tta=false
python train_bin_v6.py \
    --exp_name $exp_name \
    --data_root $data_root \
    --batch_size $batch_size \
    --val_batch_size $val_batch_size \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay \
    --backbone $backbone \
    --model $model 
    # --pretrain_from $pretrain_from
