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
module load libs/cuda

conda deactivate

conda activate semantic-decoding

echo "Job is using $SLURM_GPUS GPU(s) with ID(s) $CUDA_VISIBLE_DEVICES and $SLURM_NTASKS CPU core(s)"

python3 -c "import torch;
print(f'torch cuade is available: {torch.cuda.is_available()}')"

python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-2 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-2 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-2 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-2 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-2 --reference alpha

mv scores scores_alpha_ref_ims_S1

python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-2 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-2 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-2 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-2 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-2 --reference bravo

mv scores scores_bravo_ref_ims_S1

python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-2 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-2 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-2 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-2 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-2 --reference charlie

mv scores scores_charlie_ref_ims_S1

python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-2 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-2 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-2 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-2 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-2 --reference delta

mv scores scores_delta_ref_ims_S1

python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task alpha_repeat-2 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task bravo_repeat-2 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task charlie_repeat-2 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task delta_repeat-2 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S1 --experiment imagined_speech --task echo_repeat-2 --reference echo

mv scores scores_echo_ref_ims_S1

python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-1 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-2 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-2 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-2 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-2 --reference alpha
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-2 --reference alpha

mv scores scores_alpha_ref_ims_S3

python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-1 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-2 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-2 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-2 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-2 --reference bravo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-2 --reference bravo

mv scores scores_bravo_ref_ims_S3

python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-1 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-2 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-2 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-2 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-2 --reference charlie
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-2 --reference charlie

mv scores scores_charlie_ref_ims_S3

python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-1 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-2 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-2 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-2 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-2 --reference delta
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-2 --reference delta

mv scores scores_delta_ref_ims_S3

python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-1 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task alpha_repeat-2 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task bravo_repeat-2 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task charlie_repeat-2 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task delta_repeat-2 --reference echo
python3 decoding/evaluate_predictions.py --null 200 --subject S3 --experiment imagined_speech --task echo_repeat-2 --reference echo

mv scores scores_echo_ref_ims_S3

conda deactivate
