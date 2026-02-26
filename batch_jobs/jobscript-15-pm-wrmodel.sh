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

python3 decoding/run_decoder.py --subject S1 --word-rate-voxels speech --experiment perceived_movie --task sintel --tag wr_speech
python3 decoding/run_decoder.py --subject S1 --word-rate-voxels speech --experiment perceived_movie --task presto --tag wr_speech
python3 decoding/run_decoder.py --subject S1 --word-rate-voxels speech --experiment perceived_movie --task partlycloudy --tag wr_speech
python3 decoding/run_decoder.py --subject S1 --word-rate-voxels speech --experiment perceived_movie --task laluna --tag wr_speech
python3 decoding/run_decoder.py --subject S2 --word-rate-voxels speech --experiment perceived_movie --task sintel --tag wr_speech
python3 decoding/run_decoder.py --subject S2 --word-rate-voxels speech --experiment perceived_movie --task presto --tag wr_speech
python3 decoding/run_decoder.py --subject S2 --word-rate-voxels speech --experiment perceived_movie --task partlycloudy --tag wr_speech
python3 decoding/run_decoder.py --subject S2 --word-rate-voxels speech --experiment perceived_movie --task laluna --tag wr_speech
python3 decoding/run_decoder.py --subject S3 --word-rate-voxels speech --experiment perceived_movie --task sintel --tag wr_speech
python3 decoding/run_decoder.py --subject S3 --word-rate-voxels speech --experiment perceived_movie --task presto --tag wr_speech
python3 decoding/run_decoder.py --subject S3 --word-rate-voxels speech --experiment perceived_movie --task partlycloudy --tag wr_speech
python3 decoding/run_decoder.py --subject S3 --word-rate-voxels speech --experiment perceived_movie --task laluna --tag wr_speech
python3 decoding/run_decoder.py --subject S1 --word-rate-voxels auditory --experiment perceived_movie --task sintel --tag wr_auditory
python3 decoding/run_decoder.py --subject S1 --word-rate-voxels auditory --experiment perceived_movie --task presto --tag wr_auditory
python3 decoding/run_decoder.py --subject S1 --word-rate-voxels auditory --experiment perceived_movie --task partlycloudy --tag wr_auditory
python3 decoding/run_decoder.py --subject S1 --word-rate-voxels auditory --experiment perceived_movie --task laluna --tag wr_auditory
python3 decoding/run_decoder.py --subject S2 --word-rate-voxels auditory --experiment perceived_movie --task sintel --tag wr_auditory
python3 decoding/run_decoder.py --subject S2 --word-rate-voxels auditory --experiment perceived_movie --task presto --tag wr_auditory
python3 decoding/run_decoder.py --subject S2 --word-rate-voxels auditory --experiment perceived_movie --task partlycloudy --tag wr_auditory
python3 decoding/run_decoder.py --subject S2 --word-rate-voxels auditory --experiment perceived_movie --task laluna --tag wr_auditory
python3 decoding/run_decoder.py --subject S3 --word-rate-voxels auditory --experiment perceived_movie --task sintel --tag wr_auditory
python3 decoding/run_decoder.py --subject S3 --word-rate-voxels auditory --experiment perceived_movie --task presto --tag wr_auditory
python3 decoding/run_decoder.py --subject S3 --word-rate-voxels auditory --experiment perceived_movie --task partlycloudy --tag wr_auditory
python3 decoding/run_decoder.py --subject S3 --word-rate-voxels auditory --experiment perceived_movie --task laluna --tag wr_auditory

python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment perceived_movie  --tag wr_speech --task sintel
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment perceived_movie  --tag wr_speech --task presto
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment perceived_movie  --tag wr_speech --task partlycloudy
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment perceived_movie  --tag wr_speech --task laluna
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie  --tag wr_speech --task sintel
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie  --tag wr_speech --task presto
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie  --tag wr_speech --task partlycloudy
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie  --tag wr_speech --task laluna
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie  --tag wr_speech --task sintel
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie  --tag wr_speech --task presto
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie  --tag wr_speech --task partlycloudy
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie  --tag wr_speech --task laluna
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment perceived_movie  --tag wr_auditory --task sintel
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment perceived_movie  --tag wr_auditory --task presto
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment perceived_movie  --tag wr_auditory --task partlycloudy
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment perceived_movie  --tag wr_auditory --task laluna
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie  --tag wr_auditory --task sintel
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie  --tag wr_auditory --task presto
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie  --tag wr_auditory --task partlycloudy
python3 decoding/evaluate_predictions.py --null 200 --subject S2 --experiment perceived_movie  --tag wr_auditory --task laluna
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie  --tag wr_auditory --task sintel
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie  --tag wr_auditory --task presto
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie  --tag wr_auditory --task partlycloudy
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment perceived_movie  --tag wr_auditory --task laluna

module purge
conda deactivate

