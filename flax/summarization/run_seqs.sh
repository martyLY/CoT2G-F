model_dir=MODEL_DIR
new_model_dir=NEW_MODEL_DIR
train_data=PATH_TO_TRAIN_DATA
vaild_data=PATH_TO_VALID_DATA
test_data=PATH_TO_TEST_DATA

python run_summarization_flax.py \
	--output_dir ${new_model_dir} \
	--model_name_or_path ${model_dir} \
	--tokenizer_name ${model_dir} \
	--train_file=${train_data} \
	--test_file=${test_data} \
	--validation_file=${vaild_data} \
	--do_train --do_eval \
	--num_train_epochs 4 \
	--learning_rate 5e-5 --warmup_steps 0 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--overwrite_output_dir \
	--max_source_length 1280 \
    --max_target_length 1280 > ./seqs.log 2>&1 &