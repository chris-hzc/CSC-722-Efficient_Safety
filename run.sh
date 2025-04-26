
for norm in L2 L1 Huber MCP
do
    for attack in fgsm pgd
    do
        for model in vit swin beit convit
        do
            CUDA_VISIBLE_DEVICES=1 python main.py --model $model --norm $norm --gamma 4.0 --delta 9.0 --epsilon 1e-2 --L 3 --batch_size 8 --attack $attack &
        done
    done
done


