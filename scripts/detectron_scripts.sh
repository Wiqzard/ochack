
python -u main.py \
    --is_training 1 \
    #--resume
    --root_path ./data/ \
    --ignore_redundant \
    --partion_single_assets \
    --ratio 0.9 \
    --area_threshold 3000 \
    --model "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" \
    --num_workers 4 \
    --ims_per_batch 2 \
    --base_lr 0.0001 \
    --max_iter 3000 \
    --batch_per_img 512 \
    --eval_period 100 \
