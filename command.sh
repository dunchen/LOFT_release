# shell commands for using LoFT

# Pretraining
python3 filterwise_IST_imagenet_2xresnet18.py --rank=0 --cuda-id=0 --resume=0 &
python3 filterwise_IST_imagenet_2xresnet18.py --rank=1 --cuda-id=1 --resume=0 &
python3 filterwise_IST_imagenet_2xresnet18.py --rank=2 --cuda-id=2 --resume=0 &
python3 filterwise_IST_imagenet_2xresnet18.py --rank=3 --cuda-id=3 --resume=0 &

# Pruning
python prune.py --model './**checkpoint_folder**/**checkpoint_name**.pth' --save './**prune_result_folder**'

# Fine-tuning
python finetune.py --refine './**prune_result_folder**/pruned.pth.tar' --save './**target_folder**' --epochs 90

# to resume, use
python finetune.py --refine './**prune_result_folder**/pruned.pth.tar' --save './**target_folder**' --epochs 90 --resume './**target_folder**/apoint.pth.tar'
