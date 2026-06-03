#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem-per-gpu={mem}m
#SBATCH --mail-user=limcaoco@umich.edu
#SBATCH --output={log_file}

# Load conda environment
#source ~/opt/anaconda3/etc/profile.d/conda.sh
#conda activate {conda_env}
module load cuda/12.8.1

# Set working directory
cd /home/limcaoco/turbo/limcaoco/Boltz_affinity

# Use turbo filesystem for .boltz cache (avoids home quota pressure)
export BOLTZ_CACHE=/home/limcaoco/turbo/limcaoco/.boltz

# Run affinity prediction
echo "Starting affinity prediction at $(date)"
echo "Receptor: {receptor}"
echo "Ligands:  {ligands}"
echo "Output:   {output}"

boltz rescore receptor \
  --receptor {receptor} \
  --ligands {ligands} \
  --output {output} \
  --output-format {output_format} \
  --sort-by {sort_by} \
  --log-level {log_level}{msa_flag}

echo "Finished at $(date)"
