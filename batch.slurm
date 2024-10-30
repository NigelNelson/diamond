#SBATCH --job-name=surgical_copilot_v1a_3g
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --partition=interactive
#SBATCH --account=healthcareeng_holoscan

# Set the necessary variables
WANDB_MODE=offline

# Create the logs directory if it doesn't exist
mkdir -p logs

# Run the training script inside the container
srun \
  --container-image="nvcr.io/nvidian/onboarding/fremy-vila-1.5:latest" \
  --container-mounts="/lustre/fsw/portfolios/healthcareeng/users/nigeln:/lustre/fsw/portfolios/healthcareeng/users/nigeln,/lustre/fsw/portfolios/healthcareeng/projects/healthcareeng_holoscan/surgicalNgen/JIGSAW_data:/JIGSAW_data" \
  bash -c "
  cd /lustre/fsw/portfolios/healthcareeng/users/nigeln/surgicalNgen/diamond && \
  git pull && \
  pip install -r requirements.txt && \
  python run_cluster.py
  "
