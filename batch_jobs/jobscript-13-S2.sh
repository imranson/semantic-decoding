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

module load libs/cuda
conda activate semantic-decoding

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

python3 -c "import torch;
print(f'torch cuade is available: {torch.cuda.is_available()}')"

python3 decoding/run_decoder.py --subject S2 --experiment perceived_speech --task wheretheressmoke --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task sintel --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task presto --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task partlycloudy --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task laluna --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-1 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-2 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-1 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-2 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-1 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-2 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-1 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-2 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-1 --width 100 --tag width_100
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-2 --width 100 --tag width_100

python3 decoding/run_decoder.py --subject S2 --experiment perceived_speech --task wheretheressmoke --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task sintel --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task presto --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task partlycloudy --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task laluna --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-1 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-2 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-1 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-2 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-1 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-2 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-1 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-2 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-1 --width 200 --tag width_200
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-2 --width 200 --tag width_200

python3 decoding/run_decoder.py --subject S2 --experiment perceived_speech --task wheretheressmoke --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task sintel --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task presto --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task partlycloudy --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task laluna --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-1 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-2 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-1 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-2 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-1 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-2 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-1 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-2 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-1 --width 400 --tag width_400
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-2 --width 400 --tag width_400

python3 decoding/run_decoder.py --subject S2 --experiment perceived_speech --task wheretheressmoke --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task sintel --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task presto --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task partlycloudy --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task laluna --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-1 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-2 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-1 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-2 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-1 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-2 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-1 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-2 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-1 --width 800 --tag width_800
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-2 --width 800 --tag width_800

python3 decoding/run_decoder.py --subject S2 --experiment perceived_speech --task wheretheressmoke --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task sintel --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task presto --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task partlycloudy --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment perceived_movie --task laluna --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-1 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task alpha_repeat-2 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-1 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task bravo_repeat-2 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-1 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task charlie_repeat-2 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-1 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task delta_repeat-2 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-1 --width 1600 --tag width_1600
python3 decoding/run_decoder.py --subject S2 --experiment imagined_speech --task echo_repeat-2 --width 1600 --tag width_1600

module purge
conda deactivate
