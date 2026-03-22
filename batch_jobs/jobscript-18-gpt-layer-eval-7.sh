#!/bin/bash --login
### Choose ONE of the following partitions depending on your permitted access

#SBATCH -p gpuL              
### Required flags
#SBATCH -G 1                 # (or --gpus=N) Number of GPUs 
#SBATCH -t 4-0               # Wallclock timelimit (1-0 is one day, 4-0 is max permitted)

### Optional flags
#SBATCH -n 8          # (or --ntasks=) Number of CPU (host) cores (default is 1)
                             # See above for number of cores per GPU you can request.
                             # Also affects host RAM allocated to job unless --mem=num used.

#SBATCH --mail-type=ALL
#SBATCH --mail-user=chun.tham@student.manchester.ac.uk

module purge
conda deactivate

module load libs/cuda
conda activate semantic-decoding

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

python3 -c "import torch;
print(f'torch cuda is available: {torch.cuda.is_available()}')"

python3 decoding/run_decoder.py --subject S2 --experiment perceived_speech --task wheretheressmoke --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task presto --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task partlycloudy --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task laluna --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment perceived_multispeaker --task attend-M --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment perceived_multispeaker --task attend-F --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_speech --task wheretheressmoke --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie --task presto --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie --task partlycloudy --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie --task laluna --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_multispeaker --task attend-M --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_multispeaker --task attend-F --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task alpha_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task alpha_repeat-2 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task bravo_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task bravo_repeat-2 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task charlie_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task charlie_repeat-2 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task delta_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task delta_repeat-2 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task echo_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment imagined_speech --task echo_repeat-2 --tag gpt_layer_7

python3 decoding/run_decoder.py --subject S3 --experiment perceived_speech --task wheretheressmoke --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment perceived_movie --task presto --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment perceived_movie --task partlycloudy --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment perceived_movie --task laluna --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment perceived_multispeaker --task attend-M --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment perceived_multispeaker --task attend-F --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task alpha_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task alpha_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task bravo_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task bravo_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task charlie_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task charlie_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task delta_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task delta_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task echo_repeat-1 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/run_decoder.py --subject S3 --experiment imagined_speech --task echo_repeat-2 --gpt-layer 7 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_speech --task wheretheressmoke --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie --task presto --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie --task partlycloudy --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie --task laluna --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_multispeaker --task attend-M --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_multispeaker --task attend-F --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-2 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-2 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-2 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-2 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-1 --tag gpt_layer_7
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-2 --tag gpt_layer_7

module purge
conda deactivate

