

CUDA_VISIBLE_DEVICES=1 python main.py --model vit --norm MCP --gamma 4.0 --delta 9.0 --epsilon 1e-2 --L 3 --batch_size 8 --attack fgsm &
