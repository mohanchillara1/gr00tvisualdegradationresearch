export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache
export TMPDIR=/workspace/tmp

for NOVEL in tomato_sauce bbq_sauce ketchup milk; do
    for COND in A_clean B_dim D_blur F_diverse; do
        echo "NOVEL_OBJ - ${NOVEL} - ${COND}" | tee -a /workspace/novel_obj_results.txt
        CUDA_VISIBLE_DEVICES=0 /Isaac-GR00T/.venv/bin/python3 gr00t/eval/run_gr00t_server.py \
            --model-path /workspace/checkpoints/${COND}_v2/checkpoint-2000 \
            --embodiment-tag libero_sim \
            --use-sim-policy-wrapper &
        SERVER_PID=$!
        sleep 120
        gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
            --n-episodes 50 \
            --policy-client-host 127.0.0.1 \
            --policy-client-port 5555 \
            --max-episode-steps 720 \
            --env-name "libero_sim/pick_up_the_${NOVEL}_and_place_it_in_the_basket" \
            --n-action-steps 8 \
            --n-envs 1 2>&1 | tee -a /workspace/novel_obj_results.txt || echo "FAILED: ${NOVEL} ${COND}" | tee -a /workspace/novel_obj_results.txt
        kill $SERVER_PID 2>/dev/null
        sleep 20
    done
done