# VENCE

Code for our AAAI 2023 paper: Converge to the Truth: Factual Error Correction via Iterative Constrained Editing
![avatar](img.png)
## Dependencies
```shell
pip install -r requirements.txt
```
## Download data
Can be downloaded from this [Google Drive folder](https://drive.google.com/drive/folders/1hzwg5NtVUB_cfXiADkSanCq0JjaQ87tV). This is the FEVER intermediate annotation data released by James Thorne et al in Paper：Evidence-based Factual Error Correction
## Training the t5 model
```shell
python t5/finetune.py \
--model_name_or_path t5-base \
--do_train \
--do_eval \
--train_file $train_file \
--validation_file $valid_file \
--output_dir /home/xr/Fudan/fec_last/vence/t5/model \
--overwrite_output_dir \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--predict_with_generate \
--text_column input \
--summary_column output
```
## Training the verfication model
```shell
python verfication/finetune.py \
--model_name_or_path xlm-roberta-base \
--do_train \
--do_eval \
--train_file $train_file \
--validation_file $valid_file \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=8 \
--max_seq_length 512 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir /home/xr/Fudan/fec_last/vence/verfication/model \
--overwrite_output_dir 
```
## Running VENCE：
```shell
python main/main.py \
--iter_num 15 \
--es_lm 0.08 \
--es_ver 100 \
--es_dis 8
```
 
