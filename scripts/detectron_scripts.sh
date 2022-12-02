
python -u main.py \
    --is_training True \
    #--resume
    --root_path ./data/ \
    --ignore_redundant \
    --writer_period 100 \
    --partion_single_assets \
    --ratio 0.9 \
    --area_threshold 3000 \
    --model "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" \
    --patience 1000 \
    --num_workers 4 \
    --ims_per_batch 2 \
    --base_lr 0.0001 \
    --max_iter 3000 \
    --batch_per_img 512 \
    --patience 500 \ 
    --eval_period 100 \
    --checkpoint_period 100 \
    --use_gpu

    