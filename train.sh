# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=2

# ReferItGame
# python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch_size 8 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet101 --detr_model /home/kk/duola/TransVG-ada-dual/checkpoints/detr-r101-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50
CUDA_VISIBLE_DEVICES=6 python -u train_lt.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-referit.pth --bert_enc_num 12 --detr_enc_num 6 --dataset referit --max_query_len 20 --output_dir outputs/referit_r50


# # RefCOCO
#python -m torch.distributed.launch --nproc_per_node=4 --use_env train_lt.py --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /home/kk/.duola/ada-dual/checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6  --dataset unc --max_query_len 20  --output_dir outputs/cross-mem --epochs 130
#python -m torch.distributed.launch --nproc_per_node=4 --use_env train_lt.py --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet101 --detr_model /home/kk/.duola/ada-dual/checkpoints/detr-r101-unc.pth --bert_enc_num 12 --detr_enc_num 6 --vl_enc_layers 6 --dataset unc --max_query_len 20 --output_dir outputs/ours_101 --resume outputs/ours_101/best_checkpoint.pth --epochs 150

#CUDA_VISIBLE_DEVICES=2 python -u train_lt.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc --max_query_len 20 --output_dir outputs/coco --epochs 110 --lr_drop 60

# # RefCOCO+
#python -m torch.distributed.launch --nproc_per_node=4 --use_env train_lt.py --batch_size 16 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /home/kk/.duola/ada-dual/checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/coco+_all --epochs 180 --lr_drop 120
#CUDA_VISIBLE_DEVICES=2 python -u train_lt.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-unc.pth --bert_enc_num 12 --detr_enc_num 6 --dataset unc+ --max_query_len 20 --output_dir outputs/coco+ --epochs 180 --lr_drop 120


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=4 --use_env train_lt.py --batch_size 16 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model /home/kk/.duola/ada-dual/checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/refcocog_gsplit_r50 --epoch 180
#CUDA_VISIBLE_DEVICES=4 python -u train_lt.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref --max_query_len 40 --output_dir outputs/cocog-g 


# # RefCOCOg umd-split
#python -m torch.distributed.launch --nproc_per_node=4 --use_env train_lt.py --batch_size 16 --lr_bert 0.00001 --aug_scale --aug_translate --aug_crop --backbone resnet50 --detr_model /home/kk/.duola/ada-dual/checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/refcocog_u_all --epoch 120
#CUDA_VISIBLE_DEVICES=5 python -u train_lt.py --batch_size 64 --lr_bert 0.00001 --aug_crop --aug_scale --aug_translate --backbone resnet50 --detr_model /share/home/liuting/transvg_data/checkpoints/detr-r50-gref.pth --bert_enc_num 12 --detr_enc_num 6 --dataset gref_umd --max_query_len 40 --output_dir outputs/cocog-u
