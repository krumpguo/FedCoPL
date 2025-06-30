#!/bin/bash
TAG="exp_ID-textPT"     #exp_ID

# Function to run the job
run_job() {
    OPTIM_SEED=${optim_seed} \
    VIS_ENCODER=${vis_encoder} \
    DATASET_NAME=${dataset_name} \
    MODEL=${SETTING} \
    DATASET_DIR=${dataset_dir} \
    OUTPUT_DIR=${DIR} \
    EPOCHS=${epoch_num} \
    LR=${lr} \
    DECAY=${decay} \
    CONF_QUANTILE=${CONF_QUANTILE} \
    ENT_QUANTILE=${ENT_QUANTILE} \
    BATCH_SIZE=${batch_size} \
    TEMPERATURE=${TEMPERATURE} \
    Device_ID=${device_id} \
    Round=${round} \
    Client_Num=${client_num} \
    Partition=${partition} \
    Local_Epoch=${local_epoch} \
    Beta=${beta} \
    LR_attention=${lr_attention} \
    Num_repesudo_round=${num_repesudo_round} \
    Scheduler=${scheduler} \
    Joining_rate=${joining_rate} \
    accelerate launch --config_file methods_config/accelerate_localtest_config.yml run_main.py \
                      --model_config ${SETTING}_config_PLL.yml
}

EPOCHS=(10)
DECAY=(0.05)
BATCH_SIZE=(64)
dataset_dirs=('dataset')    #  add the path here containing datasets
vis_encoders=('ViT-B/32') #  'ViT-B/32' or 'ViT-L/14' 'ViT-B/16' 'RN50'
dataset_names=('DTD') #  'DTD' 'EuroSAT' 'RESICS45' 'Flowers102' 'cifar10' 'cifar100' 'FGVCAircraft'  
SETTINGS=('our_dual_local_prompt')   # our_dual_local_prompt
device_ids=('6') 
num_repesudo_rounds=('5') 
optim_seeds=(2)     # 1 2 3 are the kkseeds we used
round=(20) 
partition=("noniid")  #"iid" "noniid" "noniid-labeldir"
local_epoch=("10")  
betas=("0.1" )     
lrs=('0.1')   
lr_attentions=('0.5') 
TEMPERATUREs=(1.0)
scheduler=('cosine') 
joining_rates=('1') 

CONF_QUANTILEs=('50')
ENT_QUANTILEs=('50')

for dataset_dir in "${dataset_dirs[@]}"; do
for dataset_name in "${dataset_names[@]}"; do

client_num=(10)

if [ "$dataset_name" == "Flowers102" ]; then
client_num=(5)
fi

if [ "$dataset_name" == "UCF101" ]; then
client_num=(5)
fi

if [ "$dataset_name" == "CUB" ]; then
client_num=(5)
fi

if [ "$dataset_name" == "FGVCAircraft" ]; then
client_num=(5)
fi

for vis_encoder in "${vis_encoders[@]}"; do
for optim_seed in "${optim_seeds[@]}"; do

for epoch_num in "${EPOCHS[@]}"; do
for decay in "${DECAY[@]}"; do
for batch_size in "${BATCH_SIZE[@]}"; do
for SETTING in "${SETTINGS[@]}"; do

for device_id in "${device_ids[@]}"; do
for beta in "${betas[@]}"; do

for TEMPERATURE in "${TEMPERATUREs[@]}"; do
for CONF_QUANTILE in "${CONF_QUANTILEs[@]}"; do
for ENT_QUANTILE in "${ENT_QUANTILEs[@]}"; do
for num_repesudo_round in "${num_repesudo_rounds[@]}"; do
for lr_attention in "${lr_attentions[@]}"; do
for lr in "${lrs[@]}"; do
for joining_rate in "${joining_rates[@]}"; do

    LOG_FILE="script_results/log_${TAG}_${dataset_name}.txt"
    total_iterations=$((${#EPOCHS[@]} * \
                        ${#DECAY[@]} * \
                        ${#BATCH_SIZE[@]} * \
                        ${#dataset_dirs[@]} * \
                        ${#vis_encoders[@]} * \
                        ${#dataset_names[@]} * \
                        ${#SETTINGS[@]} * \
                        ${#optim_seeds[@]} * \
                        ${#methods[@]} * \
                        ${#TEMPERATUREs[@]} * \
                        ${#device_ids[@]} * \
                        ${#lrs[@]} * \
                        ${#lr_attentions[@]} * \
                        ${#num_repesudo_rounds[@]} * \
                        ${#betas[@]}))
  
    echo "The loop will iterate $total_iterations times."

    common_id="dataset-${dataset_name}_setting-${SETTING}_encoder-${vis_encoder}_split-${split_seed}_seed-${optim_seed}_epoch-${epoch_num}_lr-${lr}_decay-${decay}_bs-${batch_size}_method-${method}_T-${TEMPERATURE}_confQ-${CONF_QUANTILE}"
    DIR=./output/${dataset_name}/${SETTING}/${vis_encoder}_SplitSeed${split_seed}-${TAG}/SEED${optim_seed}/${common_id}
    
    run_job

done
done
done
done
done

done
done
done
done
done

done
done
done
done
done

done
done



