python -u train.py --model gmflownet --name gmflownet_mix-chairs --stage chairs --validation chairs --gpus 0 1 --num_steps 120000 --batch_size 10 --lr 0.0004 --image_size 368 496 --wdecay 0.0001 --use_mix_attn
python -u train.py --model gmflownet --name gmflownet_mix-things --stage things --validation sintel kitti --restore_ckpt checkpoints/gmflownet_mix-chairs.pth --gpus 0 1 --num_steps 160000 --batch_size 8 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --use_mix_attn
python -u train.py --model gmflownet --name gmflownet_mix-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/gmflownet_mix-things.pth --gpus 0 1 --num_steps 160000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma 0.85 --use_mix_attn
python -u train.py --model gmflownet --name gmflownet_mix-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/gmflownet_mix-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85 --use_mix_attn