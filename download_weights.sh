#!/bin/bash
script_dir=$(cd $(dirname $0); pwd)


cd weights
# Trained in https://github.com/Hibiki1020/attitude_estimator_in_timesformer.git
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1r4x6xUIx5PeyfdTYnIGVeIljY6UKtJRE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1r4x6xUIx5PeyfdTYnIGVeIljY6UKtJRE" -O $script_dir/weights/weights_2022_0831_timesformer.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1LFw0Bj_CxyQBHDh-awE5r3Tfv2VDSQEz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1LFw0Bj_CxyQBHDh-awE5r3Tfv2VDSQEz" -O $script_dir/weights/weights_2022_1123_timesformer.pth && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/12wG1-Mn-a-O5_mgx3-V3qissycmVx0Lx/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=12wG1-Mn-a-O5_mgx3-V3qissycmVx0Lx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=12wG1-Mn-a-O5_mgx3-V3qissycmVx0Lx" -O $script_dir/weights/weights_2022_12_08_resnet50_pretrain.pth && rm -rf /tmp/cookies.txt

# https://drive.google.com/file/d/1lVgqLxjPCYGZoJ7Fxt5rxm3JteKaah9j/view?usp=sharing
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1lVgqLxjPCYGZoJ7Fxt5rxm3JteKaah9j' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lVgqLxjPCYGZoJ7Fxt5rxm3JteKaah9j" -O $script_dir/weights/weights_2022_12_12_resnet50_pretrain_10_epoch_8frame.pth && rm -rf /tmp/cookies.txt

# Original Weights of TimeSformer
#wget -O TimeSformer_divST_8x32_224_K600.pyth "https://www.dropbox.com/s/4h2qt41m2z3aqrb/TimeSformer_divST_8x32_224_K600.pyth?dl=1"

wget https://download.pytorch.org/models/resnet18-f37072fd.pth
wget https://download.pytorch.org/models/resnet34-b627a593.pth
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
wget https://download.pytorch.org/models/resnet101-63fe2227.pth
wget https://download.pytorch.org/models/resnet152-394f9c45.pth

cd $script_dir