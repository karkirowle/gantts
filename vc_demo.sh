#!/bin/bash

# GAN-based voice cnoversion demo

set -e

experiment_id=vc_gan_CM08_M07
cmu_arctic_dir=/home/boomkin/ssh_temp/UA-Speech/audio/
#data_dir=./data/cmu_arctic_vc
data_dir=./data/vc_gan_CM08_M07
speaker_1=CM08_psola_data_augment_M07
speaker_2=M07
run_training=1

echo "Experimental id:" $experiment_id
echo "Data dir:" $data_dir
echo "CMU Arctic dir:" ${cmu_arctic_dir}

generated_audio_dir=./generated/${experiment_id}
checkpoints_dir=./checkpoints/${experiment_id}

# Feature extraction
python prepare_features_vc.py --max_files=500 ${cmu_arctic_dir} \
    $speaker_1 $speaker_2 --dst_dir=${data_dir}

# train_gan.sh args:
# 1. Hyper param name
# 2. X: Network inputs
# 3. Y: Network outputs
# 4. Where to save checkpoints
# 5. Generator wamup epoch
# 6. discriminator_warmup_epoch
# 7. Total epoch for spoofing model training
# 8. Total epoch for GAN

# Traing models
if [ "${run_training}" == 1 ]; then
    ./train_gan.sh vc ${data_dir}/X ${data_dir}/Y \
        ${checkpoints_dir} \
        50 10 50 200 $experiment_id
fi

### Evaluation ###

# Baseline
python evaluation_vc.py \
    ${checkpoints_dir}/baseline/checkpoint_epoch200_Generator.pth \
    ${data_dir} \
    ${cmu_arctic_dir}/${speaker_1} \
    ${generated_audio_dir}/baseline \
    --diffvc

# GAN
python evaluation_vc.py \
    ${checkpoints_dir}/gan/checkpoint_epoch200_Generator.pth \
    ${data_dir} \
    ${cmu_arctic_dir}/${speaker_1} \
    ${generated_audio_dir}/gan \
    --diffvc
