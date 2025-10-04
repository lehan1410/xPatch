export CUDA_VISIBLE_DEVICES=3

model_name=TimeXer

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1

python3 -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset \
  --data_path exchange_rate.csv \
  --model_id exchange_rate_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 512 \
  --d_ff 512 \
  --des 'Exp' \
  --batch_size 16 \
  --learning_rate 0.001 \
  --itr 1
