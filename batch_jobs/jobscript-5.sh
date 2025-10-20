#!/bin/bash --login
### Choose ONE of the following partitions depending on your permitted access

#SBATCH -p gpuA              
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

conda activate semantic-decoding

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

python3 -c "import torch;
print(f'torch cuade is available: {torch.cuda.is_available()}')"

python3 decoding/evaluate_predictions.py --subject S3 --experiment perceived_speech --task wheretheressmoke

python3 decoding/evaluate_predictions.py --subject S3 --experiment perceived_movie --task sintel
python3 decoding/evaluate_predictions.py --subject S3 --experiment perceived_movie --task presto
python3 decoding/evaluate_predictions.py --subject S3 --experiment perceived_movie --task partlycloudy
python3 decoding/evaluate_predictions.py --subject S3 --experiment perceived_movie --task laluna

python3 decoding/evaluate_predictions.py --subject S3 --experiment perceived_multispeaker --task attend-M
python3 decoding/evaluate_predictions.py --subject S3 --experiment perceived_multispeaker --task attend-F

python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task alpha_repeat-1
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task alpha_repeat-2
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task bravo_repeat-1
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task bravo_repeat-2
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task charlie_repeat-1
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task charlie_repeat-2
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task delta_repeat-1
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task delta_repeat-2
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task echo_repeat-1
python3 decoding/evaluate_predictions.py --subject S3 --experiment imagined_speech --task echo_repeat-2

conda deactivate
