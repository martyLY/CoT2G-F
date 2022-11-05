python run_t5_mlm_flax.py \
	--output_dir="./s-t5-small-vallina" \
	--model_type="t5-small" \
	--config_name="./s-t5-small-vallina" \
	--tokenizer_name="./s-t5-small-vallina" \
	--train_file="/data00/transed_data/S/S.txt" \
	--max_seq_length="1300" \
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
	--eval_steps="2500" > ./train_v2.log 2>&1 &