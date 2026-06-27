# gr00tvisualdegradationresearch

GR00T N1.7 Visual Degradation Research

Training Data Visual Degradation and Object Transfer in GR00T N1.7: A Systematic Study

Mohan Chillara · Aahan Kumbham · Aravind Chiruvelli

Wakeland High School, Frisco TX · Panther Creek High School, Frisco TX · Walmart Global Technology


Overview

This repository contains the code and experimental pipeline for a systematic study of how visual degradation applied to fine-tuning demonstration data affects GR00T N1.7's task success rate and novel-object generalization on the LIBERO-Object benchmark.

We fine-tuned six independent GR00T N1.7 checkpoints under five post-processing degradation conditions and evaluated each over 150 episodes (3 trials × 50 episodes). We also introduce the Object Transfer Score (OTS), a novel metric quantifying how training data visual quality conditions novel-object generalization in manipulation policies.

Paper: [______]

Key Findings

ConditionMean SROTSA_clean (baseline)0.590.017B_dim (brightness ×0.30)0.770.013C_rotation (30° tilt)0.00excludedD_blur (Gaussian blur)0.310.000E_combined (all three)0.00excludedF_diverse (50/50 mix)0.730.021


Dimmed training data outperforms clean baseline by 30.5% — counterintuitive result, three hypotheses discussed in the paper
30° rotation produces complete policy failure across all 150 evaluation episodes
OTS is near-zero universally; single-object fine-tuning does not produce novel-object transfer regardless of data quality



Repository Structure

gr00tvisualdegradationresearch/
│
├── convert_and_degrade.py          # Main pipeline: TFRecord → LeRobot + degradation
├── evaluationScriptForNovelObj.sh  # Novel object evaluation across all conditions
├── compute_image_stats.py          # Per-condition image quality metrics (future work)
│
└── README.md


Environment Setup

This code runs on RunPod (or any Ubuntu machine) with an NVIDIA A40 (48 GB VRAM). Use the PyTorch 2.1 template on RunPod — do not use PyTorch 2.8.

bash# 1. Clone Isaac-GR00T
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T && bash scripts/deployment/dgpu/install_deps.sh

# 2. Activate environment
source $HOME/.local/bin/env && source .venv/bin/activate
uv pip install -e . --no-deps
uv pip install torch==2.7.1+cu128 torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. System dependencies
apt install -y libnpp-12-8 cuda-nvcc-12-8 libegl1-mesa-dev cmake git-lfs
bash /Isaac-GR00T/gr00t/eval/sim/LIBERO/setup_libero.sh

# 4. Required config patches (run before any training)
sed -i 's/bf16: bool = True/bf16: bool = False/' [path]/training_config.py
sed -i 's/tf32: bool = True/tf32: bool = False/' [path]/training_config.py
sed -i 's/modality_keys=\["image"\]/modality_keys=["image","wrist_image"]/' [path]/embodiment_configs.py
sed -i 's/save_only_model: bool = False/save_only_model: bool = True/' [path]/finetune_config.py

# 5. Re-export every new session
export PYTHONPATH=/Isaac-GR00T:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0


Running the Pipeline

Step 1 — Convert and degrade the dataset

bashpython3 convert_and_degrade.py

This reads all 44 alphabet soup episodes from the LIBERO TFRecord dataset and produces six degraded dataset folders at /workspace/datasets/{A_clean, B_dim, C_rotation, D_blur, E_combined, F_diverse}/.

Requires the dataset at /workspace/datasets/libero. Download via:

bashpython3 -c "import tensorflow_datasets as tfds; tfds.load('libero_object_no_noops', data_dir='/workspace/datasets/libero')"

Step 2 — Fine-tune each condition

bashfor COND in A_clean B_dim C_rotation D_blur E_combined F_diverse; do
    python scripts/gr00t_finetune.py \
        --dataset-path /workspace/datasets/${COND} \
        --output-dir /workspace/checkpoints/${COND}_v2 \
        --embodiment-tag libero_sim \
        --num-steps 2000 \
        --batch-size 8
done

Training configuration: AdamW optimizer, lr=1e-4, cosine annealing decay, weight decay=1e-4, gradient clipping norm=1.0.

Step 3 — Evaluate training object SR

bashfor COND in A_clean B_dim C_rotation D_blur E_combined F_diverse; do
    # Start policy server
    CUDA_VISIBLE_DEVICES=0 python gr00t/eval/run_gr00t_server.py \
        --model-path /workspace/checkpoints/${COND}_v2/checkpoint-2000 \
        --embodiment-tag libero_sim \
        --use-sim-policy-wrapper &

    sleep 120  # Required: wait for checkpoint shards to load

    # Run evaluation client
    python gr00t/eval/rollout_policy.py \
        --n-episodes 50 \
        --env-name "libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket" \
        --n-action-steps 8 \
        --n-envs 1

    kill %1
    sleep 20
done

Step 4 — Evaluate novel object OTS

bashbash evaluationScriptForNovelObj.sh

Evaluates all four eligible conditions (A_clean, B_dim, D_blur, F_diverse) against four novel objects: tomato_sauce, bbq_sauce, ketchup, milk.


Critical Bug Fixes

Two bugs caused 0% success rate on all conditions before being identified and resolved. Document for anyone reproducing this work:

Bug 1 — Gripper Action Convention Inversion

LIBERO TFRecords store gripper values in [-1, +1]. The evaluation pipeline applies normalize_gripper_action then invert_gripper_action, expecting [0, 1] input. Without the fix, every grasp is inverted.

python# Apply at data conversion time in convert_and_degrade.py
def fix_gripper(actions):
    actions = actions.copy()
    actions[..., -1] = (-actions[..., -1] + 1.0) / 2.0
    return actions

# Verification:
# Closed (-1.0): (-(-1)+1)/2 = 1.0 → normalize → +1 → invert → -1 (closes) ✓
# Open   (+1.0): (-(+1)+1)/2 = 0.0 → normalize → -1 → invert → +1 (opens)  ✓

Bug 2 — Missing Wrist Camera Modality

The initial conversion extracted only the main camera. GR00T N1.7 requires both camera streams. Fix: extract observation.images.wrist_image during conversion and declare both in modality.json.


Citation

If you use this code or find this work useful, please cite:

bibtex@article{chillara2026groot,
  author  = {Chillara, Mohan and Kumbham, Aahan and Chiruvelli, Aravind},
  title   = {Training Data Visual Degradation and Object Transfer in {GR00T} {N1.7}: A Systematic Study},
  journal = {arXiv preprint},
  year    = {2026}
}

(Update with arXiv ID and RA-L DOI when available)


License

MIT License. See LICENSE file.


Acknowledgments

NVIDIA for open-sourcing the Isaac-GR00T framework and GR00T N1.7 weights. The LIBERO team at UNC Chapel Hill for the benchmark and dataset. Dr. Alice E. Smith (Auburn University) for methodological guidance. Compute resources via RunPod cloud infrastructure.
