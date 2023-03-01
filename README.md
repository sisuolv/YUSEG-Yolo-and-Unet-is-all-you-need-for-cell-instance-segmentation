### inference

```
python custom_det.py --img 1280 --source patched_cache --weights runs/fold_4.pt runs/fold_3.pt runs/fold_2.pt runs/fold_1.pt runs/fold_0.pt --name testa --max-det 20000 --half --iou-thres 0.5 --conf-thres=0.4 --save-txt --save-conf --line-thickness 1 --hide-labels --project patched_cache/detect --nosave
```
