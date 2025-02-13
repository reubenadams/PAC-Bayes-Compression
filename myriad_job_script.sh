# Force bash as the executing shell.
#$ -S /bin/bash

# Wallclock time (format hours:minutes:seconds).
#$ -l h_rt=6:0:0

# Request memory. Check in Process Memory in Use (MB) in wandb. Actual memory is num cores x this value (because smp 8 below).
#$ -l mem=1G

# -l gpu=1  # Change to #$ if you want to use GPUs.
#$ -pe smp 4  # Number of cores requested.

# Only launch on A100 nodes.
# -ac allow=L

# Set the name of the job.
#$ -N pac_bayes_compression

#$ -R y  # Reserve resources once they become free, until you have as much as you need. You definitely want this.
#$ -j y  # Merge the error and output streams into a single file (stdout, stderr). You definitely want this.

#$ -cwd  # All scripts have to work in a directory. This line says to use the directory we launched the script in.

date  # Print the current time.
# nvidia-smi  # Prints the GPUs available.

cd $HOME/PAC-Bayes-Compression  # Get this by running pwd in terminal.

module load python/3.11.4
source .venv/bin/activate

# This is software you don't need.
# source $UCL_CONDA_PATH/etc/profile.d/conda.sh
# conda activate stackingbert

# module load beta-modules
# module unload gcc-libs/4.9.2
# module load gcc-libs/10.2.0
# module unload compilers/intel/2018/update3
# module load compilers/gnu/10.2.0

# export HF_DATASETS_CACHE=$HOME/Scratch/hf_dataset_cache  # Put datasets on scratch space. You're not using this as your datasets are small.
mkdir $TMPDIR/wandb_cache  # Put wandb data on tmp as wandb writes lots of small files. This might speed up the job, and is polite.
export WANDB_CACHE_DIR=$TMPDIR/wandb_cache
mkdir $TMPDIR/wandb
export WANDB_DIR=$TMPDIR/wandb
export WANDB_API_KEY=$(head -n 1 $HOME/PAC-Bayes-Compression/wandb_api_key.txt)  # Setting the API key for wandb.

# wandb agent daguerro/cramming-pretrain/ted60tg3 --count 20  # Count is the number of runs to do.
python distillation_base_training.py