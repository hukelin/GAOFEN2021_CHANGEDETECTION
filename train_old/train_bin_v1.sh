exp_name="CD_v2"
data_root="/home/kelin/code/GaoFen2021_ChangeDetection/datasets/data_CD"
batch_size=4
val_batch_size=4
test_batch_size=4
epochs=50
lr=0.0001
weight_decay=0.0001
backbone="resnet50"
model="ChangeStarFarSeg"
# lightweight=false
# pretrain_from="/home/kelin/code/GaoFen2021_ChangeDetection/exp_result/ChangeStarFarSeg_resnet50_CD_v2_2021-10-05-10:04/checkpoints/ChangeStarFarSeg_resnet50_bin_49.10.pth"
# pretrained=false
# tta=false
# save_mask=false
# use_pseudo_label=false
# --tta \
# --save_mask \
# --pretrained \
python train_bin_v1.py \
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
