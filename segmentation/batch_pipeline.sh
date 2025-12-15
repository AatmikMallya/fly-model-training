#!/bin/bash

#SBATCH --job-name=mt_pipeline
#SBATCH --output=logs/pipeline_%A.out
#SBATCH --error=logs/pipeline_%A.err
#SBATCH --partition=scavenge
#SBATCH --time=9:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-user=aatmik.mallya@yale.edu
#SBATCH --mail-type=FAIL
#SBATCH --requeue
#SBATCH --open-mode=append

# Set up environment
module load Python/3.10.8-GCCcore-12.2.0
source /home/am3833/jupyterlab_venv/bin/activate

# Directory setup
BASE_DIR="/home/am3833/project"
mkdir -p $BASE_DIR/{preprocess_output,unet_output,final_output,logs,state}
STATE_DIR="$BASE_DIR/state"

# Function to check stage completion
is_complete() {
    local bodyid=$1
    local stage=$2
    [[ -f "$STATE_DIR/${bodyid}_${stage}_complete" ]]
}

# Function to mark stage as complete
mark_complete() {
    local bodyid=$1
    local stage=$2
    touch "$STATE_DIR/${bodyid}_${stage}_complete"
}

# Function to submit a job and return its job ID
submit_job() {
    local script=$1
    local bodyid=$2
    local stage=$3
    
    # Check if already complete
    if is_complete "$bodyid" "$stage"; then
        echo "Stage $stage already complete for bodyId $bodyid"
        return 0
    fi
    
    # Create stage-specific job script
    local job_script="$STATE_DIR/${bodyid}_${stage}_job.sh"
    cat > "$job_script" << 'EOL'
#!/bin/bash
EOL
    
    # Add stage-specific SLURM directives
    case $stage in
        "preprocess")
            cat >> "$job_script" << 'EOL'
#SBATCH --partition=scavenge
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --open-mode=append
EOL
            ;;
            
        "inference")
            cat >> "$job_script" << 'EOL'
#SBATCH --partition=scavenge_gpu
#SBATCH --time=2:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --constraint=ampere
#SBATCH --requeue
#SBATCH --open-mode=append
EOL
            ;;
            
        "postprocess")
            cat >> "$job_script" << 'EOL'
#SBATCH --partition=scavenge
#SBATCH --time=3:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --open-mode=append
EOL
            ;;
    esac
    
    # Add common setup and command
    cat >> "$job_script" << EOL
module load Python/3.10.8-GCCcore-12.2.0
source /home/am3833/jupyterlab_venv/bin/activate

# If this is a retry, wait briefly
if [ "\$SLURM_RESTART_COUNT" -gt 0 ]; then
    sleep \$(( RANDOM % 60 + 1 ))
fi

EOL
    
    # Add stage-specific command
    case $stage in
        "preprocess")
            echo "python $BASE_DIR/${script}.py --bodyId $bodyid --output_dir $BASE_DIR/preprocess_output" >> "$job_script"
            ;;
        "inference")
            echo "python $BASE_DIR/${script}.py --input_dir $BASE_DIR/preprocess_output --output_dir $BASE_DIR/unet_output --checkpoint_path $BASE_DIR/best_final_model.pt --bodyId $bodyid" >> "$job_script"
            ;;
        "postprocess")
            echo "python $BASE_DIR/${script}.py --input_dir $BASE_DIR/unet_output --output_dir $BASE_DIR/final_output --bodyId $bodyid" >> "$job_script"
            ;;
    esac
    
    # Add completion marking
    cat >> "$job_script" << EOL

# Mark completion on success
if [ \$? -eq 0 ]; then
    touch "$STATE_DIR/${bodyid}_${stage}_complete"
    # Clean up intermediate files after successful postprocessing
    if [ "$stage" = "postprocess" ]; then
        rm -f $BASE_DIR/preprocess_output/*_${bodyid}.npy
        rm -f $BASE_DIR/unet_output/*_${bodyid}.npy
    fi
fi
EOL
    
    chmod +x "$job_script"
    sbatch --parsable \
        --job-name="${stage}_${bodyid}" \
        --output="logs/${stage}_%j.out" \
        --error="logs/${stage}_%j.err" \
        "$job_script"
}

# Function to wait for a job
wait_for_job() {
    local job_id=$1
    local status
    
    while true; do
        status=$(squeue -h -j "$job_id" -o "%t")
        if [[ -z "$status" ]]; then
            # Job not in queue, check if it completed successfully
            sacct -j "$job_id" -o State -n | grep -q "COMPLETED"
            return $?
        elif [[ "$status" == "PD" || "$status" == "R" || "$status" == "CG" ]]; then
            sleep 30
        else
            # Job failed or was cancelled
            return 1
        fi
    done
}

# Track progress
echo "0" > "$STATE_DIR/current_bodyid_index"

# Process each bodyId
while read -r bodyid; do
    current_index=$(cat "$STATE_DIR/current_bodyid_index")
    echo "Processing bodyId: $bodyid (index: $current_index)"
    
    # Submit and monitor each stage
    for stage in "preprocess" "inference" "postprocess"; do
        if ! is_complete "$bodyid" "$stage"; then
            job_id=$(submit_job "$stage" "$bodyid" "$stage")
            
            if [[ -n "$job_id" ]]; then
                if ! wait_for_job "$job_id"; then
                    echo "Stage $stage failed for bodyId $bodyid"
                    exit 1
                fi
            fi
        fi
    done
    
    echo "$((current_index + 1))" > "$STATE_DIR/current_bodyid_index"
    echo "Completed processing bodyId $bodyid"
done < "$BASE_DIR/lc_bodyids.txt"

echo "Pipeline completed"
rm -f "$STATE_DIR/current_bodyid_index"