model_dir=MODEL_DIR
data=PATH_TO_DATA
max_seq_length=MAX_SEQ_LENGTH

python run_t5_mlm_flax.py \
	--output_dir=${MODEL_DIR} \
	--model_type="t5-small" \
	--config_name=${MODEL_DIR} \
	--tokenizer_name=${MODEL_DIR} \
	--train_file=${data} \
	--max_seq_length=${MAX_SEQ_LENGTH} \
	--num_train_epochs="5" \
	--per_device_train_batch_size="16" \
	--per_device_eval_batch_size="16" \
	--adafactor \
	--do_train \
	--do_eval \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="1000" \
	--eval_steps="2500" > ./pre-train.log 2>&1