#iou=0.5 0.9039779
#conf=0.4

iou=0.5 #0.9039779
conf=0.4

rm -r runs/detect/fold_*
# Multi-GPU
for i in 0 1 2 3 4; do
  #sleep $(expr 30 \* $i) &&  # 30-second delay (optional)
  echo 'Starting Fold '$i'...' &&
  nohup python detect.py --img 1280 --source ../patch_cv/fold_$i/valid --weights runs/train/fold_$i/weights/best.pt --name fold_$i --max-det 20000 --half --iou-thres $iou --conf-thres $conf --save-txt --save-conf --line-thickness 1 --hide-labels > infer_fold_$i.log &
done
rm ../cv_pred/*


#python detect.py --img 1280 --source ../patch_cv/fold_0/valid --weights runs/train/fold_0/weights/best.pt --name fold_0 --max-det 20000 --half --iou-thres 0.5 --conf-thres=0.25 --save-txt --save-conf --line-thickness 1 --hide-labels
#python detect.py --img 1280 --source ../patch_cv/fold_1/valid --weights runs/train/fold_1/weights/best.pt --name fold_1 --max-det 20000 --half --iou-thres 0.5 --conf-thres=0.25 --save-txt --save-conf --line-thickness 1 --hide-labels
#python detect.py --img 1280 --source ../patch_cv/fold_2/valid --weights runs/train/fold_2/weights/best.pt --name fold_2 --max-det 20000 --half --iou-thres 0.5 --conf-thres=0.25 --save-txt --save-conf --line-thickness 1 --hide-labels
#python detect.py --img 1280 --source ../patch_cv/fold_3/valid --weights runs/train/fold_3/weights/best.pt --name fold_3 --max-det 20000 --half --iou-thres 0.5 --conf-thres=0.25 --save-txt --save-conf --line-thickness 1 --hide-labels
#python detect.py --img 1280 --source ../patch_cv/fold_4/valid --weights runs/train/fold_4/weights/best.pt --name fold_4 --max-det 20000 --half --iou-thres 0.5 --conf-thres=0.25 --save-txt --save-conf --line-thickness 1 --hide-labels