export CUDA_VISIBLE_DEVICES=1


python -u run.py \
  --is_training 1 \
  --model_id faster_rcnn_train1 \
  --model Faster_RCNN\
  --data custom \
  --root_path ./data/ \
  --ratio 0.8 \
  --data_path SORDI_2022_Single_Assets \
  --checkpoints ./checkpoints/ \
  --batch_size 3 \
  --num_workers 0 \
  --itr 1 \
  --train_epochs 2 \
  --patience 7 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --loss 'mse' \
  --lradj 'type1' \
  --use_gpu True \
  --gpu 0 \
  --devices 0



  --use_multi_gpu False \
  --use_amp False \