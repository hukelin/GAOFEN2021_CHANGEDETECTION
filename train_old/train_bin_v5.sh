exp_name="TrainLEVIR_CD_v5"
data_root="/home/kelin/data/LEVIR-CD"
batch_size=4
val_batch_size=4
test_batch_size=4
epochs=50
lr=0.001
weight_decay=0.0001
backbone="hrnet_w30"
model="ChangeModel"
# lightweight=false
# pretrain_from="exp_result/ChangeModel_hrnet_w30_TrainLEVIR_CD_v2_2021-10-11-13:34/checkpoints/ChangeModel_hrnet_w30_bin_44.94.pth"
# pretrained=false
# tta=false
# save_mask=false
# use_pseudo_label=false
# --tta \
# --save_mask \
# --pretrained \
python train_bin_v5.py \
    --exp_name $exp_name \
    --data_root $data_root \
    --batch_size $batch_size \
    --val_batch_size $val_batch_size \
    --test_batch_size $test_batch_size \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay \
    --backbone $backbone \
    --model $model 
    # --pretrain_from $pretrain_from
