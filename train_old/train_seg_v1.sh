exp_name="Seg_v1"
data_root="/home/kelin/data/train"
batch_size=4
val_batch_size=4
test_batch_size=4
epochs=60
lr=0.0001
weight_decay=0.0001
backbone="resnet50"
model="farseg"
# lightweight=false
pretrain_from="exp_result/farseg_resnet50_Seg_v1_2021-10-04-14:11/checkpoints/farseg_resnet50_seg_5.42.pth"
# pretrained=false
# tta=false
# save_mask=false
# use_pseudo_label=false
# --tta \
# --save_mask \
# --pretrained \
python train_seg_v1.py \
    --exp_name $exp_name \
    --data_root $data_root \
    --batch_size $batch_size \
    --val_batch_size $val_batch_size \
    --test_batch_size $test_batch_size \
    --epochs $epochs \
    --lr $lr \
    --weight_decay $weight_decay \
    --backbone $backbone \
    --model $model \
    --pretrain_from $pretrain_from
