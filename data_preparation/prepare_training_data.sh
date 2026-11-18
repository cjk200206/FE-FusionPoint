#!/bin/bash
# GPU stage: generate SuperPoint + SuperGlue pseudo-labels, one sequence at a time.
set -e

usage() {
    echo "Usage: bash $0 -d RAW_DATASET_DIR [-t OUTPUT_ROOT]"
    echo "Run prepare_training_data_parallel.sh first; split_data.py is a separate step."
}

data_dir=""
target_dir=""
while getopts "d:t:h" flag; do
    case "$flag" in
        d) data_dir=$OPTARG ;;
        t) target_dir=$OPTARG ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 1 ;;
    esac
done

if [[ ! -d "$data_dir" ]]; then
    usage >&2
    exit 1
fi
data_dir=${data_dir%/}
dataset_name=${data_dir##*/}
target_dir=${target_dir:-$(dirname -- "$data_dir")/SuperEvent_data}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob
sequences=("$data_dir"/*/)
if [[ ${#sequences[@]} -eq 0 ]]; then
    echo "No sequence directories found in $data_dir" >&2
    exit 1
fi

for dir in "${sequences[@]}"; do
    dir=${dir%/}
    seq=${dir##*/}
    sequence_target_dir=""
    for split in "" train val test; do
        candidate="$target_dir/${split:+$split/}$dataset_name/$seq"
        if [[ -d "$candidate/frames" ]]; then
            sequence_target_dir=$candidate
            break
        fi
    done
    if [[ -z "$sequence_target_dir" ]]; then
        echo "Prepared frames missing for $dataset_name/$seq; run the CPU stage first." >&2
        exit 1
    fi

    if [[ -d "$sequence_target_dir/sg_matches" ]] && \
       [[ "$dataset_name" != gopro || -d "$sequence_target_dir/low_res_sg_matches" ]]; then
        echo "Skipping $dataset_name/$seq: pseudo-labels already exist."
        continue
    fi
    echo "Generating pseudo-labels for $dataset_name/$seq"
    python "$repo_root/data_preparation/prepare_pseudo_groundtruth.py" "$sequence_target_dir"
done

echo "Pseudo-label generation complete. Run split_data.py after preparing all datasets."
