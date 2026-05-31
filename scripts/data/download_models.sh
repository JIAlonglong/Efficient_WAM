#!/bin/bash
# Download Motus pretrained models via hf-mirror.com, bypassing proxy.
# Usage:
#   bash scripts/data/download_models.sh          # download all
#   bash scripts/data/download_models.sh qwen      # download Qwen3-VL only
#   bash scripts/data/download_models.sh motus     # download Motus only
#   bash scripts/data/download_models.sh robotwin  # download Motus_robotwin2 only

set -euo pipefail

MIRROR="https://hf-mirror.com"
BASE_DIR="$(cd "$(dirname "$0")/../.." && pwd)/pretrained_models"

# Bypass proxy
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy 2>/dev/null || true

download_file() {
    local url="$1"
    local output="$2"
    local size_hint="$3"

    if [ -f "$output" ]; then
        local existing_size=$(stat -c%s "$output" 2>/dev/null || echo 0)
        echo "  [exists] $output ($(( existing_size / 1048576 )) MB)"
        return 0
    fi

    echo "  [download] $output ($size_hint)"
    mkdir -p "$(dirname "$output")"
    wget -c -q --show-progress --no-proxy "$url" -O "$output" 2>&1 | tail -1
    echo "  [done] $output"
}

download_qwen() {
    echo ""
    echo "=========================================="
    echo "Downloading Qwen/Qwen3-VL-2B-Instruct"
    echo "=========================================="
    local dir="$BASE_DIR/Qwen3-VL-2B-Instruct"
    local url_prefix="$MIRROR/Qwen/Qwen3-VL-2B-Instruct/resolve/main"

    # Small files first
    for f in .gitattributes README.md chat_template.json config.json generation_config.json merges.txt preprocessor_config.json tokenizer.json tokenizer_config.json video_preprocessor_config.json vocab.json; do
        download_file "$url_prefix/$f" "$dir/$f" "small"
    done

    # Large model file
    download_file "$url_prefix/model.safetensors" "$dir/model.safetensors" "4.3 GB"
}

download_motus() {
    echo ""
    echo "=========================================="
    echo "Downloading motus-robotics/Motus"
    echo "=========================================="
    local dir="$BASE_DIR/Motus"
    local url_prefix="$MIRROR/motus-robotics/Motus/resolve/main"

    for f in .gitattributes LICENSE README.md config.json; do
        download_file "$url_prefix/$f" "$dir/$f" "small"
    done

    download_file "$url_prefix/mp_rank_00_model_states.pt" "$dir/mp_rank_00_model_states.pt" "16 GB"
}

download_robotwin() {
    echo ""
    echo "=========================================="
    echo "Downloading motus-robotics/Motus_robotwin2"
    echo "=========================================="
    local dir="$BASE_DIR/Motus_robotwin2"
    local url_prefix="$MIRROR/motus-robotics/Motus_robotwin2/resolve/main"

    for f in .gitattributes LICENSE README.md config.json; do
        download_file "$url_prefix/$f" "$dir/$f" "small"
    done

    download_file "$url_prefix/mp_rank_00_model_states.pt" "$dir/mp_rank_00_model_states.pt" "16 GB"
}

# Parse args
if [ $# -eq 0 ]; then
    echo "Usage: $0 [qwen|motus|robotwin|all]"
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        qwen)     download_qwen ;;
        motus)    download_motus ;;
        robotwin) download_robotwin ;;
        all)      download_qwen; download_motus; download_robotwin ;;
        *)        echo "Unknown: $arg (use qwen, motus, robotwin, or all)"; exit 1 ;;
    esac
done

echo ""
echo "All downloads complete!"
