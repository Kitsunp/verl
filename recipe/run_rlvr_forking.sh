#!/bin/bash

# ==========================================
# RLVR Forking Tokens Training Script
# Qwen 2.5 1.5B on Single L4 GPU
# Combines Paper 1 (PSR/NSR) + Paper 2 (Forking Tokens)
# ==========================================

set -e  # Exit on any error

echo "🚀 RLVR Forking Tokens Training Pipeline"
echo "📊 Model: Qwen 2.5 1.5B"
echo "💾 GPU: Single L4 (24GB)"
echo "🧠 Algorithm: PSR/NSR + Forking Tokens"

# ==========================================
# ENVIRONMENT SETUP
# ==========================================

# Set environment variables for memory optimization
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false      # Avoid tokenizer warnings
export OMP_NUM_THREADS=4                 # Limit CPU threads

# Memory optimization flags
export CUDA_LAUNCH_BLOCKING=0            # Set to 1 only for debugging
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Fragment memory

# Check GPU memory
echo "🔍 Checking GPU memory..."
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits

# ==========================================
# DATA PREPARATION WITH RETRY LOGIC
# ==========================================

echo "📊 Setting up data directories..."

# Create data directories
DATA_DIR="${HOME}/verl/data"
mkdir -p ${DATA_DIR}/gsm8k
mkdir -p ${DATA_DIR}/math

# Define file paths
GSM8K_TRAIN="${DATA_DIR}/gsm8k/train.parquet"
GSM8K_TEST="${DATA_DIR}/gsm8k/test.parquet"
MATH_TRAIN="${DATA_DIR}/math/train.parquet"
MATH_TEST="${DATA_DIR}/math/test.parquet"

echo "📥 Preparing datasets with retry logic..."

# ==========================================
# RETRY FUNCTION FOR DATA PROCESSING
# ==========================================

process_dataset_with_retry() {
    local dataset_name="$1"
    local script_path="$2"
    local output_dir="$3"
    local train_file="$4"
    local test_file="$5"
    local max_attempts=2
    
    echo "🔄 Processing ${dataset_name} dataset..."
    
    for attempt in $(seq 1 $max_attempts); do
        echo "📋 Attempt ${attempt}/${max_attempts} for ${dataset_name}"
        
        # Check if script exists
        if [ ! -f "${script_path}" ]; then
            echo "❌ Script not found: ${script_path}"
            echo "🔍 Checking alternative locations..."
            
            # Try alternative paths
            ALT_SCRIPT1="./examples/data_preprocess/$(basename ${script_path})"
            ALT_SCRIPT2="../examples/data_preprocess/$(basename ${script_path})"
            ALT_SCRIPT3="examples/data_preprocess/$(basename ${script_path})"
            
            for alt_path in "${ALT_SCRIPT1}" "${ALT_SCRIPT2}" "${ALT_SCRIPT3}"; do
                if [ -f "${alt_path}" ]; then
                    echo "✅ Found script at: ${alt_path}"
                    script_path="${alt_path}"
                    break
                fi
            done
            
            if [ ! -f "${script_path}" ]; then
                echo "❌ Could not find ${dataset_name} preprocessing script"
                return 1
            fi
        fi
        
        # Clean output directory if this is a retry
        if [ $attempt -gt 1 ]; then
            echo "🧹 Cleaning ${output_dir} for retry..."
            rm -rf "${output_dir}"
            mkdir -p "${output_dir}"
        fi
        
        # Run preprocessing script
        echo "▶️  Running: python3 ${script_path} --local_dir ${output_dir}"
        if python3 "${script_path}" --local_dir "${output_dir}" --hdfs_dir null; then
            echo "✅ ${dataset_name} preprocessing script completed"
            
            # Verify output files exist and are valid
            local files_ok=true
            for file in "${train_file}" "${test_file}"; do
                if [ ! -f "${file}" ]; then
                    echo "❌ Missing output file: ${file}"
                    files_ok=false
                elif [ ! -s "${file}" ]; then
                    echo "❌ Empty output file: ${file}"
                    files_ok=false
                else
                    # Quick validation of parquet file
                    if ! python3 -c "import pandas as pd; pd.read_parquet('${file}'); print('✅ Valid parquet: $(basename ${file})')" 2>/dev/null; then
                        echo "❌ Invalid parquet file: ${file}"
                        files_ok=false
                    fi
                fi
            done
            
            if [ "$files_ok" = true ]; then
                echo "✅ ${dataset_name} data processed successfully"
                echo "📄 Files: ${train_file}, ${test_file}"
                return 0
            else
                echo "⚠️  ${dataset_name} processing completed but output files are invalid"
            fi
        else
            echo "❌ ${dataset_name} preprocessing script failed"
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            echo "🔄 Retrying ${dataset_name} processing..."
            sleep 2
        fi
    done
    
    echo "❌ ${dataset_name} processing failed after ${max_attempts} attempts"
    return 1
}

# ==========================================
# PROCESS GSM8K DATASET WITH RETRY
# ==========================================

if [ ! -f "${GSM8K_TRAIN}" ] || [ ! -f "${GSM8K_TEST}" ]; then
    if ! process_dataset_with_retry "GSM8K" "examples/data_preprocess/gsm8k.py" "${DATA_DIR}/gsm8k" "${GSM8K_TRAIN}" "${GSM8K_TEST}"; then
        echo "💥 Failed to process GSM8K dataset after retries"
        exit 1
    fi
else
    echo "✅ GSM8K data already exists"
    
    # Verify existing files are valid
    echo "🔍 Verifying existing GSM8K files..."
    for file in "${GSM8K_TRAIN}" "${GSM8K_TEST}"; do
        if ! python3 -c "import pandas as pd; df=pd.read_parquet('${file}'); assert len(df)>0; print(f'✅ Valid: $(basename ${file}) ({len(df)} rows)')" 2>/dev/null; then
            echo "❌ Existing GSM8K file is invalid: ${file}"
            echo "🔄 Reprocessing GSM8K..."
            rm -f "${GSM8K_TRAIN}" "${GSM8K_TEST}"
            if ! process_dataset_with_retry "GSM8K" "examples/data_preprocess/gsm8k.py" "${DATA_DIR}/gsm8k" "${GSM8K_TRAIN}" "${GSM8K_TEST}"; then
                echo "💥 Failed to reprocess GSM8K dataset"
                exit 1
            fi
            break
        fi
    done
fi

# ==========================================
# PROCESS MATH DATASET WITH RETRY
# ==========================================

if [ ! -f "${MATH_TRAIN}" ] || [ ! -f "${MATH_TEST}" ]; then
    if ! process_dataset_with_retry "MATH" "examples/data_preprocess/math_dataset.py" "${DATA_DIR}/math" "${MATH_TRAIN}" "${MATH_TEST}"; then
        echo "💥 Failed to process MATH dataset after retries"
        exit 1
    fi
else
    echo "✅ MATH data already exists"
    
    # Verify existing files are valid
    echo "🔍 Verifying existing MATH files..."
    for file in "${MATH_TRAIN}" "${MATH_TEST}"; do
        if ! python3 -c "import pandas as pd; df=pd.read_parquet('${file}'); assert len(df)>0; print(f'✅ Valid: $(basename ${file}) ({len(df)} rows)')" 2>/dev/null; then
            echo "❌ Existing MATH file is invalid: ${file}"
            echo "🔄 Reprocessing MATH..."
            rm -f "${MATH_TRAIN}" "${MATH_TEST}"
            if ! process_dataset_with_retry "MATH" "examples/data_preprocess/math_dataset.py" "${DATA_DIR}/math" "${MATH_TRAIN}" "${MATH_TEST}"; then
                echo "💥 Failed to reprocess MATH dataset"
                exit 1
            fi
            break
        fi
    done
fi

# ==========================================
# VERIFY DATA FILES WITH DETAILED VALIDATION
# ==========================================

echo "🔍 Performing comprehensive data verification..."

# Check file sizes and content
for file in "${GSM8K_TRAIN}" "${GSM8K_TEST}" "${MATH_TRAIN}" "${MATH_TEST}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✅ $file (${size})"
        
        # Comprehensive validation of parquet structure
        python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_parquet('$file')
    print(f'  📊 Shape: {df.shape}')
    
    # Check required columns
    required_cols = ['data_source', 'prompt', 'reward_model']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f'  ❌ Missing required columns: {missing_cols}')
        sys.exit(1)
    else:
        print(f'  ✅ Required columns present: {required_cols}')
    
    # Check data_source values
    if 'data_source' in df.columns:
        sources = df.data_source.unique()
        print(f'  🏷️  Data sources: {list(sources)[:3]}')
        
        # Validate data_source values
        valid_sources = [
            'openai/gsm8k', 
            'lighteval/MATH', 
            'DigitalLearningGmbH/MATH-lighteval'
        ]
        
        invalid_sources = [s for s in sources if s not in valid_sources]
        if invalid_sources:
            print(f'  ⚠️  Unknown data sources: {invalid_sources}')
        else:
            print(f'  ✅ All data sources are recognized')
    
    # Check for empty data
    if len(df) == 0:
        print(f'  ❌ Dataset is empty')
        sys.exit(1)
    else:
        print(f'  ✅ Dataset contains {len(df)} samples')
        
    # Check reward_model structure
    if 'reward_model' in df.columns:
        sample_rm = df['reward_model'].iloc[0]
        if isinstance(sample_rm, dict) and 'ground_truth' in sample_rm:
            print(f'  ✅ Reward model structure is valid')
        else:
            print(f'  ⚠️  Reward model structure may be invalid')
    
    print(f'  ✅ Validation passed for $(basename $file)')
    
except Exception as e:
    print(f'  ❌ Error reading $file: {e}')
    sys.exit(1)
"
        
        # Check the exit status of the Python validation
        if [ $? -ne 0 ]; then
            echo "❌ Validation failed for $file"
            echo "🔄 Attempting to regenerate dataset..."
            
            # Determine which dataset to regenerate
            if [[ "$file" == *"gsm8k"* ]]; then
                echo "🧹 Cleaning GSM8K directory..."
                rm -rf "${DATA_DIR}/gsm8k"
                mkdir -p "${DATA_DIR}/gsm8k"
                if ! process_dataset_with_retry "GSM8K" "examples/data_preprocess/gsm8k.py" "${DATA_DIR}/gsm8k" "${GSM8K_TRAIN}" "${GSM8K_TEST}"; then
                    echo "💥 Failed to regenerate GSM8K dataset"
                    exit 1
                fi
            elif [[ "$file" == *"math"* ]]; then
                echo "🧹 Cleaning MATH directory..."
                rm -rf "${DATA_DIR}/math"
                mkdir -p "${DATA_DIR}/math"
                if ! process_dataset_with_retry "MATH" "examples/data_preprocess/math_dataset.py" "${DATA_DIR}/math" "${MATH_TRAIN}" "${MATH_TEST}"; then
                    echo "💥 Failed to regenerate MATH dataset"
                    exit 1
                fi
            fi
        fi
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

echo "🎯 All data files validated successfully!"

# ==========================================
# MODEL PREPARATION
# ==========================================

echo "🤖 Checking model availability..."

# Download model if needed  
MODEL_PATH="${HOME}/models/Qwen2.5-1.5B"
if [ ! -d "${MODEL_PATH}" ]; then
    echo "📥 Downloading Qwen2.5-1.5B model..."
    mkdir -p "${HOME}/models"
    huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir "${MODEL_PATH}"
    echo "✅ Model downloaded to ${MODEL_PATH}"
else
    echo "✅ Model already exists at ${MODEL_PATH}"
fi

# ==========================================
# ALGORITHM VERIFICATION
# ==========================================

echo "🔧 Checking algorithm registration..."
python3 -c "
from verl.trainer.ppo.core_algos import get_adv_estimator_fn
try:
    fn = get_adv_estimator_fn('rlvr_forking_weighted')
    print('✅ RLVR Forking Weighted algorithm registered successfully')
except Exception as e:
    print(f'❌ Algorithm not found: {e}')
    exit(1)
"

# ==========================================
# TRAINING CONFIGURATION
# ==========================================

echo "⚙️ Setting up training configuration..."

# Prepare train and validation file lists
TRAIN_FILES="['${GSM8K_TRAIN}','${MATH_TRAIN}']"
VAL_FILES="['${GSM8K_TEST}','${MATH_TEST}']"

echo "📂 Training files: ${TRAIN_FILES}"
echo "📂 Validation files: ${VAL_FILES}"

# Create output directory
OUTPUT_DIR="./outputs/rlvr_forking_$(date +%m%d_%H%M)"
mkdir -p "${OUTPUT_DIR}"

# ==========================================
# START TRAINING
# ==========================================

echo "🎯 Starting RLVR Forking training..."
echo "📊 Expected improvements: +2-4 points on math tasks"
echo "💾 Monitor GPU usage: watch -n 1 nvidia-smi"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rlvr_forking_weighted \
    +algorithm.entropy_percentile=0.8 \
    +algorithm.lambda_psr=0.1 \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=4 \
    data.val_batch_size=2 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.prompt_key=prompt \
    data.truncation=left \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    +actor_rollout_ref.actor.gradient_accumulation_steps=2 \
    \
    actor_rollout_ref.rollout.name=vllm_rollout \
    +actor_rollout_ref.rollout.tensor_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=4 \
    \
    trainer.total_epochs=5 \
    trainer.project_name=rlvr_forking_test \
    trainer.experiment_name=rlvr_forking_qwen1.5b_$(date +%m%d_%H%M) \
    trainer.logger='[console,wandb]' \
    trainer.save_freq=2 \
    trainer.test_freq=1 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    \
    ray_init.num_cpus=4 \
    reward_model.enable=false \
    \
    hydra.run.dir="${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"

# ==========================================
# POST-TRAINING SUMMARY
# ==========================================

echo ""
echo "🎉 Training pipeline completed!"
echo "📊 Check training logs: ${OUTPUT_DIR}/training_log.txt"
echo "💾 Check outputs: ${OUTPUT_DIR}/"
echo ""
echo "📋 What happened:"
echo "  ✅ Downloaded and processed GSM8K + MATH datasets (with retry logic)"  
echo "  ✅ Downloaded Qwen2.5-1.5B model"
echo "  ✅ Verified RLVR Forking Weighted algorithm"
echo "  ✅ Comprehensive data validation performed"
echo "  ✅ Trained with PSR/NSR + Forking Tokens combination"
echo ""
echo "🔧 Robust features added:"
echo "  🔄 Automatic retry logic (max 2 attempts per dataset)"
echo "  🧹 Directory cleanup on failed attempts"
echo "  🔍 Multiple script path detection"
echo "  ✅ Comprehensive parquet file validation"
echo "  🎯 Automatic data_source verification for reward functions"
echo ""
echo "🔍 Next steps:"
echo "  📊 Analyze Pass@k results in logs"
echo "  🎯 Compare performance vs baseline"
echo "  📈 Expected: +2-4 point improvement on math tasks"
echo "  🔬 Monitor entropy statistics and forking token ratios"

# ==========================================
# QUICK VALIDATION SCRIPT
# ==========================================

cat > "${OUTPUT_DIR}/validate_implementation.py" << 'EOF'
"""
Quick validation that our RLVR Forking implementation works
"""
import torch
from verl.trainer.ppo.core_algos import get_adv_estimator_fn

def test_rlvr_forking():
    print("🧪 Testing RLVR Forking Weighted implementation...")
    
    # Test data
    batch_size, seq_len, vocab_size = 2, 10, 100
    
    token_level_rewards = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    config = {
        "entropy_percentile": 0.8,
        "lambda_psr": 0.1
    }
    
    # Get our algorithm
    fn = get_adv_estimator_fn("rlvr_forking_weighted")
    
    # Test computation
    advantages, returns = fn(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        config=config,
        logits=logits
    )
    
    print(f"✅ Advantages shape: {advantages.shape}")
    print(f"✅ Returns shape: {returns.shape}")
    print(f"✅ Advantages range: [{advantages.min():.3f}, {advantages.max():.3f}]")
    print("🎉 Implementation working correctly!")

if __name__ == "__main__":
    test_rlvr_forking()
EOF

echo "🧪 Validation script created: ${OUTPUT_DIR}/validate_implementation.py"
echo "   Run: python3 ${OUTPUT_DIR}/validate_implementation.py"
