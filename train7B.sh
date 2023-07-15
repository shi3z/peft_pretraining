# 2xA6000 7B
torchrun --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_7b.json \
    --batch_size 1 \
    --total_batch_size 8 \
    --lr 1e-3 \
    --max_length 512 \
    --use_peft \
    --relora 5000 \
    --cycle_length 5000 \
    --restart_warmup_steps 100 \
    --scheduler cosine_restarts \
    --warmup_steps 500 \
    --reset_optimizer_on_relora True \
    --num_training_steps 20000 \
    --save_every 5000 \
    --eval_every 5000 \
    --tags relora_7b
