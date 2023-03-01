#python train.py --img 1280 --batch 8 --epochs 50 --data ../fold_0.yaml --weights yolov5s6.pt --multi-scale --seed 42 --name fold_0
python train.py --img 1280 --batch 8 --epochs 50 --data ../fold_1.yaml --weights yolov5s6.pt --multi-scale --seed 42 --name fold_1
python train.py --img 1280 --batch 8 --epochs 50 --data ../fold_2.yaml --weights yolov5s6.pt --multi-scale --seed 42 --name fold_2
python train.py --img 1280 --batch 8 --epochs 50 --data ../fold_3.yaml --weights yolov5s6.pt --multi-scale --seed 42 --name fold_3
python train.py --img 1280 --batch 8 --epochs 50 --data ../fold_4.yaml --weights yolov5s6.pt --multi-scale --seed 42 --name fold_4
