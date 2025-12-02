CUDA_VISIBLE_DEVICES=1 python tracking/test.py mcitrack mcitrack_b224 --dataset lasot --threads 2
CUDA_VISIBLE_DEVICES=1 python tracking/test.py mcitrack mcitrack_b224 --dataset trackingnet --threads 2
CUDA_VISIBLE_DEVICES=1 python tracking/test.py mcitrack mcitrack_b224 --dataset uav --threads 2