exp_name="change_v5"
data_root="/home/kelin/code/GaoFen2021_ChangeDetection/datasets/data_CD"
batch_size=4
val_batch_size=4
test_batch_size=4
epochs=60
lr=0.00001
weight_decay=0.0001
backbone="resnet50"
model="farseg"
# lightweight=false
# pretrain_from="exp_result/farseg_resnet50_change_v1_2021-10-01-05:48/checkpoints/farseg_resnet50_bin_50.03.pth"
# pretrained=false
# tta=false
# save_mask=false
# use_pseudo_label=false
# --tta \
# --save_mask \
# --pretrained \
python train_bin.py \
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
