#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

s1_predict=./data/input/s1_seq_to_seq.csv
s1_modelfile=./model/seqs_s1
s1_result=./data/output/s1_results.csv

s2_predict=./data/input/s2_seq_to_seq.csv
s2_modelfile=./model/seqs_s2
s2_result=./data/output/s2_results.csv

s_co_result=./data/output/s_co_results.csv

# s_predict=./data/input/s_seq_to_seq.csv
# s_modelfile=./model/seqs_s
# s_vallina_result=./data/output/s_vallina_results.csv

python ./flax/summarization/predict_summarization_flax.py \
	--output_dir ./seqs_s1_predict \
	--model_name_or_path ${s1_modelfile}\
	--tokenizer_name ${s1_modelfile} \
	--test_file=${s1_predict} \
	--validation_file=${s1_predict} \
	--baseline_fname=./transformers/examples/flax/summarization/spike_reference.fa \
	--protein=s1 \
	--generate_file_name=${s1_result} \
	--do_predict --predict_with_generate\
	--overwrite_output_dir \
	--per_device_eval_batch_size 64 \
	--max_source_length 1400 \
	--max_target_length 695 > ./predict_s1_seqs.log 2>&1 

wait

python ./flax/summarization/predict_summarization_flax.py \
	--output_dir ./seqs_s2_predict \
	--model_name_or_path ${s2_modelfile} \
	--tokenizer_name ${s2_modelfile} \
	--test_file=${s2_predict} \
	--validation_file=${s2_predict} \
	--baseline_fname=./transformers/examples/flax/summarization/spike_reference.fa \
	--protein=s2 \
	--generate_file_name=${s2_result} \
	--do_predict --predict_with_generate\
	--overwrite_output_dir \
	--per_device_eval_batch_size 64 \
	--max_source_length 1200 \
	--max_target_length 590 > ./predcit_s2_seqs.log 2>&1 

wait

python ./data/output/postprocess_data.py \
	--s1-filename=${s1_result} \
	--s2-filename=${s2_result} \
	--s-results-filename=${s_co_result} > ./merge.log 2>&1 




# python ./flax/summarization/predict_summarization_flax.py \
# 	--output_dir ./seqs_s_predict \
# 	--model_name_or_path ${s_modelfile} \
# 	--tokenizer_name ${s_modelfile} \
# 	--test_file=${s_predict} \
# 	--validation_file=${s_predict} \
# 	--protein=s \
# 	--baseline_fname=./transformers/examples/flax/summarization/spike_reference.fa \
# 	--do_predict --predict_with_generate\
# 	--generate_file_name ${s_vallina_result} \
# 	--overwrite_output_dir \
# 	--grammer False \
# 	--csc False \
# 	--per_device_eval_batch_size 64 \
# 	--max_source_length 1300 \
# 	--max_target_length 1300 > ./predict_s_seqs.log 2>&1
