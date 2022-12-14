python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/dataset_1.yaml --img 1280 720 --cfg cfg/training/yolov7-e6e.yaml --weights ''--name yolov7-e6e-model_1 --hyp data/hyp.scratch.p6.yaml --epochs 150 --cache-images

python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/dataset_2.yaml  --img 1280 720 --cfg cfg/training/yolov7-e6e.yaml --weights ''--name yolov7-e6e-model_2 --hyp data/hyp.scratch.p6.yaml --epochs 150 --cache-images

python train_aux.py --workers 8 --device 0 --batch-size 32 --data data/dataset_3.yaml  --img 1280 720 --cfg cfg/training/yolov7-e6e.yaml --weights ''--name yolov7-e6e-model_2 --hyp data/hyp.scratch.p6.yaml --epochs 150 --cache-images




python train.py --workers 8 --device 0 --batch-size 32 --data data/dataset_1.yaml --img 1280 720 --cfg cfg/training/yolov7x.yaml --weights ''--name yolov7-X-model_1 --hyp data/hyp.scratch.p5.yaml --epochs 150 --cache-images

python train.py --workers 8 --device 0 --batch-size 32 --data data/dataset_2.yaml  --img 1280 720 --cfg cfg/training/yolov7x.yaml--weights ''--name yolov7-X-model_2 --hyp data/hyp.scratch.p5.yaml --epochs 150 --cache-images

python train.py --workers 8 --device 0 --batch-size 32 --data data/dataset_3.yaml  --img 1280 720 --cfg cfg/training/yolov7x.yaml --weights ''--name yolov7-X-model_2 --hyp data/hyp.scratch.p5.yaml --epochs 150 --cache-images
