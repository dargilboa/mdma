#!/bin/sh


#SBATCH --mail-type=ALL
#SBATCH --mail-user=dargilboa@gmail.com
#SBATCH --job-name=BNAF_scan   # The job name.
#SBATCH -c 4                     # The number of cpu cores to use.
#SBATCH --time=1-0:0:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=8gb        # The memory the job will use per cpu core.
#SBATCH --gres=gpu:gtx1080:1     # The number of GPUs (1) and the variety (gtx1080)

cd MDMA/neural_copulas

screen -dmS 0 bash -c "module load anaconda3-2019.03; module load cuda/9.2.88; source activate ari-env; pip install tensorboardx; bash run_BNAF.sh .45 knn 0; exec sh"
#screen -dmS 1 bash -c "module load anaconda3-2019.03; module load cuda/9.2.88; source activate ari-env; pip install tensorboardx; bash run_BNAF.sh .5 knn 1; exec sh"
#screen -dmS 2 bash -c "module load anaconda3-2019.03; module load cuda/9.2.88; source activate ari-env; pip install tensorboardx; bash run_BNAF.sh .55 knn 2; exec sh"
#screen -dmS 3 bash -c "module load anaconda3-2019.03; module load cuda/9.2.88; source activate ari-env; pip install tensorboardx; bash run_BNAF.sh .6 knn 3; exec sh"
#screen -dmS 4 bash -c "module load anaconda3-2019.03; module load cuda/9.2.88; source activate ari-env; pip install tensorboardx; bash run_BNAF.sh .65 knn 4; exec sh"
#screen -dmS 5 bash -c "module load anaconda3-2019.03; module load cuda/9.2.88; source activate ari-env; pip install tensorboardx; bash run_BNAF.sh .7 knn 5; exec sh"
#screen -dmS 6 bash -c "module load anaconda3-2019.03; module load cuda/9.2.88; source activate ari-env; pip install tensorboardx; bash run_BNAF.sh .75 knn 6; exec sh"
#screen -dmS 7 bash -c "module load anaconda3-2019.03; module load cuda/9.2.88; source activate ari-env; pip install tensorboardx; bash run_BNAF.sh .8 knn 7; exec sh"
