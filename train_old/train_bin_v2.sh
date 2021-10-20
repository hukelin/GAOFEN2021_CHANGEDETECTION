exp_name="CD_v2"
data_root="/home/kelin/code/GaoFen2021_ChangeDetection/datasets/data_CD"
batch_size=4
val_batch_size=4
test_batch_size=4
epochs=60
lr=0.00001
weight_decay=0.0001
backbone="hrnet_w48"
model="fcn"
# lightweight=false
# pretrain_from="exp_result/fcn_hrnet_w48_CD_v2_2021-10-04-15:39/checkpoints/fcn_hrnet_w48_bin_25.49.pth"
# pretrained=false
# tta=false
# save_mask=false
# use_pseudo_label=false
# --tta \
# --save_mask \
# --pretrained \
python train_bin_v2.py \
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
