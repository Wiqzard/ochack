
python -u main.py \
    --is_training True \
    --test False \
    --model_checkpoint "" \
    #--resume
    --root_path ./data/ \
    --ignore_redundant \
    --partion_single_assets 3\
    --ratio 0.95 \
    --area_threshold_min 6000 \
    --area_threshold_max 700000 \

    --model "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml" \
    --nms_threshold 0.1 \
    --num_workers 4 \
    --ims_per_batch 2 \
    --gamma 0.1 \
    --base_lr 0.001 \
    --warmup_steps 1000\
    --max_iter 200000\
    --batch_per_img 512 \
    --use_gpu \
    --use_amp \
    --checkpoint_period 2000 \
    --eval_period 5000 \


    --writer_period 100 \
    --patience 1000 \


   