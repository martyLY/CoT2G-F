# CoT2G-F: Co-attention-based Transformer model to bridge Genotype and Fitness

Source code of our paper **CoT2G-F**: Co-attention-based Transformer model to bridge Genotype and Fitness, using the JAX/FAX backend.

![model architecture](https://github.com/martyLY/CoT2G-F/blob/74ebc200ee71739cae045b3d1afb2ad84b2e8cb5/figure.png)

Our method is implemented based on the open-source toolkit [Tansformers](https://huggingface.co/docs/transformers/index).

## Dependencies

- Python==3.8.8

- Jax==0.3.15

  - Running on CPU

    ```bash
    pip install --upgrade pip
    pip install --upgrade "jax[cpu]"
    ```

  - Running on single or multiple GPUs (**recommend**)

    ```bash
    pip install --upgrade pip
    # Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
    # Note: wheels only available on linux.
    pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    ```

  You can also follow this [guide for installing JAX on GPUs](https://github.com/google/jax/#pip-installation-gpu-cuda) since the installation depends on your CUDA and CuDNN version.
  

We provide a script to install all necessary python depandencies for our code. 

```bash
pip install -r ./requirements.txt
```

## Quick Start

An easy and quick way to :star2: generate the mutated spike protein sequence  on the next month time slice,  :star2: calculate the csc (conditional semantic change) score and grammeratility of related generated spike protein sequence.

We provide two ways to generate mutated spike protein sequence and calculate correlation csc score and grammeratility value.

- **CoT2G-F**: You need to input the protein sequences of a certain month (T), and the spatial neighbor sequence of each protein sequence in the previous month (T-1).
- **Vallina Transformer**: You only need to input the protein sequences of a certain month (T).

### Checkpoints

First, you should download our pre-trained checkpoints, all checkpoints are provided in this [Google drive folder](https://drive.google.com/drive/folders/1Hbm6PluF3ko6hogIz7B8HXGVotF4pztw?usp=share_link), please place the pretrained checkpoints under the directory `./model`. If you only want to do generation task, then you just need to download these three checkpoints:  `./model/seqs_s1` ,  `./model/seqs_s2`  and  `./model/seqs_s`.

### Data pre-processing

We use the public GISAID ptotein sequence dataset (download [here](https://gisaid.org)), alternatively, if you only want to inference mutations of a small part of the spike ptotein sequence and do not want to download such a large amount of GISAID data, we also provide some example data in the directory `./data/input`  for your reference and preparation.

You need to prepare a `spike.fasta` file, where you want to inference the future mutations of these protein sequences, then if you choose the **Vallina Transformer** method, execute the followed script, your data preprocessing results are placed in `s-input-filename`.

```bash
cd ./data/input
fasta_filename=FASTA
s_filename=S
python preprocess_data.py \
    --fasta-filename ${fasta_filename} \
    --mask True \
    --s-filename ${s_filename}
```

Or you want to get more accurate mutations, you can choose our **CoT2G-F** method, in this case you would also need to prepare two additional meta files, one is `date_meta_data.csv`, which specify the period when your spike protein sequence appears (please refer to [GISAID](https://gisaid.org)), it needs to be accurate to the year-month, e.g. 2020-01, 2021-12, the other is `neighbor_meta_data.csv`, in this meta data file, you need to specify the temporal neighbor sequence of the input protein sequence in the last month(please refer this).Finally, execute

```bash
cd ./data/input
fasta_filename=FASTA
neighbor_fasta_filename=NEIGHBPR_FASTA
neighbor-metadata-filename=METADATA
s1_filename=S1
s2_filename=S2
s_filename=S
python preprocess_data.py \
    --fasta-filename ${fasta_filename} \
    --neighbor-fasta-filename ${neighbor_fasta_filename} \
    --neighbor-metadata-filename ${neighbor_metadata_filename} \
    --split True \
    --mask True \
    --s1-filename ${s1_filename} \
    --s2-filename ${s2_filename} \
    --s-filename ${s_filename}
```




### Inference

**CoT2G-F**: to generate mutated spike protein sequence and calculate correlation csc score and grammeratility value using CoT2G-F method, one could execute 

```bash
export CUDA_VISIBLE_DEVICES=0

s1_predict=./data/input/s1_seq_to_seq.csv
s1_modelfile=./model/seqs_s1
s1_result=./data/output/s1_results.csv

s2_predict=./data/input/s2_seq_to_seq.csv
s2_modelfile=./model/seqs_s2
s2_result=./data/output/s2_results.csv

s_co_result=./data/output/s_co_results.csv

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
```

- `s1_predict` and `s2_predict` is the path of spike protein sequence data file through data pre-process step.
- The, `s1_modelfile`  and  `s2_modelfile`  is the path of download model checkpoints for weights initialization,  they are also They are also pretrained tokenizer path, the  `s1_modelfile` link to `./model/seqs_s1`,  `s2_modelfile`  link to `./model/seqs_s2`.
-  `s1_result`  and `s2_result`  is the path of  split spike protein sequence results  (mutated spike protein sequence and correlation csc score and grammeratility value), `s_co_result`  is the merge results of `s1_result`  and `s2_result`, the final results are saved in `s_co_result`.

- To get the mutations of generated spike protein sequence, one need a  `baseline_fname` , in our method, we select the Wuhan spike protein, which saved in `./transformers/examples/flax/summarization/spike_reference.fa`, you can also change `baseline_fname` in script.
- `protein`: one of `s`, `s1`, `s2`, you need to specify the input spike protein type, the intact spike protein sequence or split s1 protein sequence, s2 protein sequence.

**Vallina Transformer**: Using Vallina Transformer to generate mutated spike protein sequence without the intoduction of spatiotemporal information, one could execute

```bash
export CUDA_VISIBLE_DEVICES=0

s_predict=./data/input/s_seq_to_seq.csv
s_modelfile=./model/seqs_s
s_vallina_result=./data/output/s_vallina_results.csv

python ./flax/summarization/predict_summarization_flax.py \
	--output_dir ./seqs_s_predict \
	--model_name_or_path ${s_modelfile} \
	--tokenizer_name ${s_modelfile} \
	--test_file=${s_predict} \
	--validation_file=${s_predict} \
	--protein=s \
	--baseline_fname=./transformers/examples/flax/summarization/spike_reference.fa \
	--do_predict --predict_with_generate\
	--generate_file_name ${s_vallina_result} \
	--overwrite_output_dir \
	--grammer False \
	--csc False \
	--per_device_eval_batch_size 64 \
	--max_source_length 1300 \
	--max_target_length 1300 > ./predict_s_seqs.log 2>&1
```

- `csc`, `grammer`: whether to calculate and save the csc score and grammeratility value to `generate_file_name`.

## Usage 
If you want to increase the amount of spike protein training or migrate our work to other proteins, you just need to go through the following three steps:

### Tokenizer

```bash
model_dir=MODEL_DIR
data=PATH_TO_DATA

python ./flax/language-modeling/tokenizer.py \
    --model-dir ${model_dir} \
    --protein-file-name ${data}
```

### Pre-train

```bash
model_dir=MODEL_DIR
data=PATH_TO_DATA
max_seq_length=MAX_SEQ_LENGTH
top_data=PATH_TO_TOP_DATA
down_data=PATH_TO_DOWN_DATA

python ./flax/language-modeling/run_t5_mlm_flax.py \
	--output_dir=${model_dir} \
	--model_type="t5-small" \
	--config_name=${model_dir} \
	--tokenizer_name=${model_dir} \
	--train_file=${data} \
	--max_seq_length=${max_seq_length} \
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
	--eval_steps="2500" 
wait

python ./flax/language-modeling/run_t5_mlm_flax.py \
	--output_dir=${model_dir} \
	--model_type="t5-small" \
	--config_name=${model_dir} \
	--tokenizer_name=${model_dir} \
	--train_file=${top_data} \
	--max_seq_length=${max_seq_length} \
	--num_train_epochs="1" \
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
	--eval_steps="2500"
wait

python ./flax/language-modeling/run_t5_mlm_flax.py \
	--output_dir=${model_dir} \
	--model_type="t5-small" \
	--config_name=${model_dir} \
	--tokenizer_name=${model_dir} \
	--train_file=${down_data} \
	--max_seq_length=${max_seq_length} \
	--num_train_epochs="1" \
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
	--eval_steps="2500"> ./pretrain.log 2>&1
```

### Fine-tune

```bash
model_dir=MODEL_DIR
new_model_dir=NEW_MODEL_DIR
train_forward_data=PATH_TO_TRAIN_FORWARD_DATA
vaild_forward_data=PATH_TO_VALID_FORWARD_DATA
test_forward_data=PATH_TO_TEST_FORWARD_DATA
train_top_data=PATH_TO_TRAIN_TOP_DATA
vaild_top_data=PATH_TO_VALID_TOP_DATA
test_top_data=PATH_TO_TEST_TOP_DATA
train_down_data=PATH_TO_TRAIN_DOWN_DATA
vaild_down_data=PATH_TO_VALID_DOWN_DATA
test_down_data=PATH_TO_TEST_DOWN_DATA


python run_summarization_flax.py \
	--output_dir ${new_model_dir} \
	--model_name_or_path ${model_dir} \
	--tokenizer_name ${model_dir} \
	--train_file=${train_forward_data} \
	--test_file=${test_forward_data} \
	--validation_file=${vaild_forward_data} \
	--do_train --do_eval \
	--num_train_epochs 4 \
	--learning_rate 5e-5 --warmup_steps 0 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--overwrite_output_dir \
	--max_source_length 512 \
  	--max_target_length 512 

wait
python run_summarization_flax.py \
	--output_dir ${new_model_dir} \
	--model_name_or_path ${model_dir} \
	--tokenizer_name ${model_dir} \
	--train_file=${train_top_data} \
	--test_file=${test_top_data} \
	--validation_file=${vaild_top_data} \
	--do_train --do_eval \
	--num_train_epochs 1 \
	--learning_rate 5e-5 --warmup_steps 0 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--overwrite_output_dir \
	--max_source_length 512 \
  	--max_target_length 512 
wait
python run_summarization_flax.py \
	--output_dir ${new_model_dir} \
	--model_name_or_path ${model_dir} \
	--tokenizer_name ${model_dir} \
	--train_file=${train_down_data} \
	--test_file=${test_down_data} \
	--validation_file=${vaild_down_data} \
	--do_train --do_eval \
	--num_train_epochs 1 \
	--learning_rate 5e-5 --warmup_steps 0 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--overwrite_output_dir \
	--max_source_length 512 \
  	--max_target_length 512 > ./seqs.log 2>&1
```

