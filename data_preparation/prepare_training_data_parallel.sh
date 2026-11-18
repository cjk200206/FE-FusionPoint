#!/bin/bash
# CPU stage: convert frames and generate MCTS in parallel by sequence.
set -e

usage() {
    echo "Usage: bash $0 -d RAW_DATASET_DIR [-t OUTPUT_ROOT] [-j JOBS] [-u]"
}

data_dir=""
target_dir=""
NUM_CORES=32
undistort=""
while getopts "d:t:j:uh" flag; do
    case "$flag" in
        d) data_dir=$OPTARG ;;
        t) target_dir=$OPTARG ;;
        j) NUM_CORES=$OPTARG ;;
        u) undistort=True ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done

if [[ ! -d "$data_dir" || ! "$NUM_CORES" =~ ^[1-9][0-9]*$ ]]; then
    usage >&2
    exit 1
fi
data_dir=${data_dir%/}
dataset_name=${data_dir##*/}
target_dir=${target_dir:-$(dirname -- "$data_dir")/SuperEvent_data}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

# Keep successful workers quiet; print Python output on failure, without log files.
run_step() {
    local output
    if output=$("$@" 2>&1); then
        return 0
    fi
    printf '%s\n' "$output" >&2
    return 1
}

prepare_sequence() {
    local sequence_dir=${1%/}
    local seq=${sequence_dir##*/}
    local sequence_target_dir="$target_dir/$dataset_name/$seq"
    local clear_delta=(0.001 0.003 0.01 0.03 0.1)
    local normal_delta=("${clear_delta[@]}")
    local heavy_delta=(0.02 0.07 0.2 0.7 2)
    local blur_dataset=false
    case "$dataset_name" in
        uevd|gen_coco|gen_coco_1)
            blur_dataset=true
            normal_delta=(0.005 0.016 0.05 0.16 0.5)
            ;;
    esac

    echo "Processing $dataset_name/$seq"
    mkdir -p "$sequence_target_dir"
    run_step python "$repo_root/data_preparation/convert_images.py" \
        "$sequence_dir" "$sequence_target_dir" "$dataset_name" \
        --delta_t "${clear_delta[@]}" --undistort "$undistort"
    run_step python "$repo_root/data_preparation/events2ts.py" \
        "$sequence_dir" "$sequence_target_dir/time_surfaces" "$dataset_name" \
        --timestamp_path "$sequence_target_dir/frame_timestamps.txt" \
        --delta_t "${normal_delta[@]}" --undistort "$undistort"
    if [[ "$blur_dataset" == true ]]; then
        run_step python "$repo_root/data_preparation/events2ts.py" \
            "$sequence_dir" "$sequence_target_dir/time_surfaces_clear" "$dataset_name" \
            --timestamp_path "$sequence_target_dir/frame_timestamps.txt" \
            --delta_t "${clear_delta[@]}" --undistort "$undistort"
        run_step python "$repo_root/data_preparation/events2ts.py" \
            "$sequence_dir" "$sequence_target_dir/time_surfaces_heavy_blur" "$dataset_name" \
            --timestamp_path "$sequence_target_dir/frame_timestamps.txt" \
            --delta_t "${heavy_delta[@]}" --undistort "$undistort"
    fi
    echo "Done $dataset_name/$seq"
}
export -f run_step prepare_sequence
export repo_root target_dir dataset_name undistort

shopt -s nullglob
sequences=("$data_dir"/*/)
if [[ ${#sequences[@]} -eq 0 ]]; then
    echo "No sequence directories found in $data_dir" >&2
    exit 1
fi

echo "CPU preprocessing: $dataset_name, ${#sequences[@]} sequences, $NUM_CORES jobs"
if printf '%s\0' "${sequences[@]}" | xargs -0 -n 1 -P "$NUM_CORES" \
    bash -eEc 'trap '\''echo "FAILED: ${1%/}" >&2'\'' ERR; prepare_sequence "$1"' _; then
    echo "All ${#sequences[@]} sequences processed."
else
    echo "CPU preprocessing failed; see the failed sequences above." >&2
    exit 1
fi
