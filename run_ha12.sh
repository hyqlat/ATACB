#!/bin/bash
# training script

#train conti class
# CFG="humanact12_cc"
# GPU_IDX=0
#python train_cc.py --gpu_index $GPU_IDX --cfg $CFG --seed 0

#train
CFG="humanact12_rnn"
CFG_CC="humanact12_cc"
GPU_IDX=0
python exp_vae_act.py --gpu_index $GPU_IDX --cfg $CFG --cfg_cc $CFG_CC --is_other_act --seed 0

#test
CFG="humanact12_rnn"
CFG_CLASS=humanact12_act_classifier
GPU_IDX=0
TH=0.01
python eval_vae_act_stats_muti_seed.py --iter 500 --nk 10 --bs 5 --num_samp 50 --num_seed 5 --stop_fn 5 --cfg $CFG --cfg_classifier $CFG_CLASS --gpu_index $GPU_IDX --threshold $TH

#visualization
# CFG="grab_rnn"
# CFG_CLASS=grab_act_classifier
# GPU_IDX=0
# TH=0.015
# python eval_vae_act_render_video.py --iter 500 --nk 10 --num_samp 50 --stop_fn 5 --cfg $CFG --cfg_classifier $CFG_CLASS --gpu_index $GPU_IDX --threshold $TH


