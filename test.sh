# export CUDA_VISIBLE_DEVICES=0,1,2,3

# export CUDA_VISIBLE_DEVICES=0
# ReferItGame
# python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ../released_models/TransVG_referit.pth --output_dir ./outputs/referit_r50


# # RefCOCO
# python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 16 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc --max_query_len 20 --eval_set testB --eval_model ./outputs/coco_50_dual-cross_share256_neck128/checkpoint0099.pth --output_dir ./outputs_test/coco_r50_64_B_lastone_epoch99
# CUDA_VISIBLE_DEVICES=7 python -u eval.py --batch_size 64 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc --max_query_len 20 --eval_set testA --eval_model ./outputs/coco/checkpoint0109.pth --output_dir ./outputs_test/coco_share256
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset referit --max_query_len 20 --eval_set test --eval_model ./outputs/referit_r50/best_checkpoint.pth --output_dir ./outputs/referit_r50
# # RefCOCO+
#python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 16 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc+ --max_query_len 20 --eval_set testB --eval_model ./outputs/coco+/best_checkpoint.pth --output_dir ./outputs_test/coco+_r50_B
# CUDA_VISIBLE_DEVICES=7 python -u eval.py --batch_size 64 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset unc+ --max_query_len 20 --eval_set testB --eval_model ./outputs/coco+/best_checkpoint.pth --output_dir ./outputs_test/coco+_b


# # RefCOCOg g-split
# python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 16 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref --max_query_len 40 --eval_set val --eval_model ./outputs/refcocog_gsplit_r50/best_checkpoint.pth --output_dir ./outputs/refcocog_gsplit_r50
# CUDA_VISIBLE_DEVICES=7 python -u eval.py --batch_size 64 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref --max_query_len 40 --eval_set val --eval_model ./outputs/cocog-g/checkpoint0079.pth --output_dir ./outputs_test/cocog-g


# # RefCOCOg u-split
# python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 16 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ./outputs/refcocog_u_all/best_checkpoint.pth --output_dir ./outputs/refcocog_usplit_r50
CUDA_VISIBLE_DEVICES=7 python -u eval.py --batch_size 64 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ./outputs/cocog-u/checkpoint0089.pth --output_dir ./outputs_test/cocog-u

#python -m torch.distributed.launch --nproc_per_node=4 --use_env eval.py --batch_size 32 --num_workers 4 --bert_enc_num 12 --detr_enc_num 6 --backbone resnet50 --dataset gref_umd --max_query_len 40 --eval_set test --eval_model ../released_models/TransVG_gref_umd.pth --output_dir ./outputs/refcocog_usplit_r50