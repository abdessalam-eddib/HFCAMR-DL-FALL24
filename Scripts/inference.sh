CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  nohup torchrun --nproc_per_node=8 inference.py  > inference.log 2>&1 &