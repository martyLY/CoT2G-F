# python run_summarization_flax.py \
# 	--output_dir ./seqs_s1 \
# 	--model_name_or_path /data00/transformers/examples/flax/language-modeling/s-t5-small \
# 	--tokenizer_name /data00/transformers/examples/flax/language-modeling/s-t5-small \
# 	--train_file=/data00/s1_seq_to_seq.csv \
# 	--test_file=/data00/s1_seq_to_seq_val.csv \
# 	--validation_file=/data00/s1_seq_to_seq_val.csv \
# 	--do_train --do_eval --do_predict --predict_with_generate \
# 	--num_train_epochs 6 \
# 	--learning_rate 5e-5 --warmup_steps 0 \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--overwrite_output_dir \
# 	--max_source_length 1400 \
#     --max_target_length 695 > ./seqs_s1.log 2>&1 &

# python run_summarization_flax.py \
# 	--output_dir ./seqs_s2 \
# 	--model_name_or_path /data00/transformers/examples/flax/language-modeling/s-t5-small \
# 	--tokenizer_name /data00/transformers/examples/flax/language-modeling/s-t5-small \
# 	--train_file=/data00/s2_seq_to_seq.csv \
# 	--test_file=/data00/s2_seq_to_seq_val.csv \
# 	--validation_file=/data00/s2_seq_to_seq_val.csv \
# 	--do_train --do_eval --do_predict --predict_with_generate \
# 	--num_train_epochs 6 \
# 	--learning_rate 5e-5 --warmup_steps 0 \
# 	--per_device_train_batch_size 8 \
# 	--per_device_eval_batch_size 8 \
# 	--overwrite_output_dir \
# 	--max_source_length 1200 \
#     --max_target_length 590 > ./seqs_s2.log 2>&1 &


python run_summarization_flax.py \
	--output_dir ./seqs_s_v2 \
	--model_name_or_path /data00/transformers/examples/flax/language-modeling/s-t5-small-vallina \
	--tokenizer_name /data00/transformers/examples/flax/language-modeling/s-t5-small-vallina \
	--train_file=/data00/seq_to_seq.csv \
	--test_file=/data00/seq_to_seq_val.csv \
	--validation_file=/data00/seq_to_seq_val.csv \
	--do_train --do_eval \
	--num_train_epochs 2 \
	--learning_rate 5e-5 --warmup_steps 0 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--overwrite_output_dir \
	--max_source_length 1280 \
    --max_target_length 1280 > ./seqs_s_v1.log 2>&1 &