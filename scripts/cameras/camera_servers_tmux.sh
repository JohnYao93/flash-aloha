#!/usr/bin/env bash
#
# Start and manage per-camera Flash Aloha camera servers in tmux windows.
#
# Examples:
#   scripts/cameras/camera_servers_tmux.sh start
#   scripts/cameras/camera_servers_tmux.sh start left overhead
#   scripts/cameras/camera_servers_tmux.sh start --cameras left,right
#   scripts/cameras/camera_servers_tmux.sh tail left
#   scripts/cameras/camera_servers_tmux.sh status
#   scripts/cameras/camera_servers_tmux.sh stop

set -Eeuo pipefail

SESSION_NAME="${CAMERA_TMUX_SESSION:-camera_servers}"
STARTUP_TIMEOUT="${CAMERA_STARTUP_TIMEOUT:-15}"
DEFAULT_CAMERAS=(left right overhead)
VALID_CAMERAS=(left right overhead)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

usage() {
    cat <<EOF
Usage:
  $0 start [left|right|overhead ...]
  $0 start --cameras left,right [--startup-timeout SECONDS] [--replace]
  $0 tail CAMERA [LINES]
  $0 status
  $0 attach
  $0 stop

Environment:
  CAMERA_TMUX_SESSION       tmux session name (default: camera_servers)
  CAMERA_STARTUP_TIMEOUT    seconds to wait for startup (default: 15)

Examples:
  $0 start
  $0 start left overhead
  $0 tail left
  $0 tail overhead 20
EOF
}

die() {
    echo -e "${RED}Error:${NC} $*" >&2
    exit 1
}

require_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        die "tmux is not installed. Install it with: sudo apt install tmux"
    fi
}

select_encoder_gpu() {
    local query_output gpu free_mib
    local selected_gpu=""
    local selected_free_mib=-1

    command -v nvidia-smi >/dev/null 2>&1 || return 1
    query_output="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null
    )" || return 1

    while IFS=',' read -r gpu free_mib; do
        gpu="${gpu//[[:space:]]/}"
        free_mib="${free_mib//[[:space:]]/}"
        [[ "$gpu" =~ ^[0-9]+$ && "$free_mib" =~ ^[0-9]+$ ]] || continue
        if (( free_mib > selected_free_mib )) \
            || { (( free_mib == selected_free_mib )) && [[ -n "$selected_gpu" ]] && (( gpu > selected_gpu )); }; then
            selected_gpu="$gpu"
            selected_free_mib="$free_mib"
        fi
    done <<< "$query_output"

    [[ -n "$selected_gpu" ]] || return 1
    printf '%s\n' "$selected_gpu"
}

is_valid_camera() {
    local requested="$1"
    local camera
    for camera in "${VALID_CAMERAS[@]}"; do
        [[ "$requested" == "$camera" ]] && return 0
    done
    return 1
}

append_camera() {
    local camera="$1"
    local existing

    [[ -n "$camera" ]] || return 0
    [[ "$camera" == "all" ]] && {
        for existing in "${DEFAULT_CAMERAS[@]}"; do
            append_camera "$existing"
        done
        return 0
    }

    is_valid_camera "$camera" || die "unknown camera '$camera' (valid: ${VALID_CAMERAS[*]})"

    for existing in "${REQUESTED_CAMERAS[@]}"; do
        [[ "$existing" == "$camera" ]] && return 0
    done
    REQUESTED_CAMERAS+=("$camera")
}

append_camera_list() {
    local raw="$1"
    local part
    local -a parts
    IFS=',' read -ra parts <<< "$raw"
    for part in "${parts[@]}"; do
        append_camera "$part"
    done
}

tmux_session_exists() {
    tmux has-session -t "$SESSION_NAME" >/dev/null 2>&1
}

tmux_target_exists() {
    local camera="$1"
    tmux list-windows -t "$SESSION_NAME" -F '#{window_name}' 2>/dev/null | grep -Fxq "$camera"
}

run_camera() {
    local camera="$1"
    local status_file="$2"
    local encoder_gpu="${3:-}"
    local status

    cd "$PROJECT_ROOT"

    echo "=== CAMERA SERVER: $camera ==="
    echo "Command: uv run src/camera_utils/camera_server.py --camera $camera --write-shared-memory"
    if [[ -n "$encoder_gpu" ]]; then
        echo "Encoder GPU: physical GPU $encoder_gpu"
    else
        echo "Encoder GPU: FFmpeg automatic selection"
    fi
    echo

    set +e
    if [[ -n "$encoder_gpu" ]]; then
        env CUDA_VISIBLE_DEVICES="$encoder_gpu" \
            uv run src/camera_utils/camera_server.py --camera "$camera" --write-shared-memory
    else
        uv run src/camera_utils/camera_server.py --camera "$camera" --write-shared-memory
    fi
    status=$?
    set -e

    echo
    echo "[camera_servers] $camera exited with status $status"
    printf '%s\n' "$status" > "$status_file"
    exit "$status"
}

start_camera_window() {
    local camera="$1"
    local status_dir="$2"
    local encoder_gpu="$3"
    local status_file="$status_dir/$camera.status"
    local command

    command="exec $(printf '%q ' "$SCRIPT_PATH" __run-camera "$camera" "$status_file" "$encoder_gpu")"
    tmux new-window -t "$SESSION_NAME" -n "$camera" -c "$PROJECT_ROOT" >/dev/null
    tmux set-window-option -t "$SESSION_NAME:$camera" remain-on-exit on >/dev/null
    tmux send-keys -t "$SESSION_NAME:$camera" "$command" C-m
}

capture_camera_output() {
    local camera="$1"
    local lines="${2:-20}"
    tmux capture-pane -p -t "$SESSION_NAME:$camera" -S "-$lines" 2>/dev/null || true
}

stop_session() {
    if tmux_session_exists; then
        tmux kill-session -t "$SESSION_NAME" >/dev/null 2>&1 || true
    fi
}

wait_for_startup() {
    local status_dir="$1"
    shift
    local cameras=("$@")
    local start_ts now elapsed camera output status_file status
    local -A ready=()

    start_ts="$(date +%s)"

    while true; do
        for camera in "${cameras[@]}"; do
            [[ "${ready[$camera]:-0}" == "1" ]] && continue

            status_file="$status_dir/$camera.status"
            if [[ -f "$status_file" ]]; then
                status="$(cat "$status_file" 2>/dev/null || true)"
                echo -e "${RED}Camera '$camera' failed to start; process exited with status ${status:-unknown}.${NC}" >&2
                echo -e "${YELLOW}Last output from '$camera':${NC}" >&2
                capture_camera_output "$camera" 25 >&2
                stop_session
                exit 1
            fi

            output="$(capture_camera_output "$camera" 200)"
            if grep -Fq "[Camera Server] Running" <<< "$output"; then
                ready[$camera]=1
                echo -e "${GREEN}[OK]${NC} $camera camera server is running"
            fi
        done

        local all_ready=1
        for camera in "${cameras[@]}"; do
            if [[ "${ready[$camera]:-0}" != "1" ]]; then
                all_ready=0
                break
            fi
        done
        [[ "$all_ready" == "1" ]] && return 0

        now="$(date +%s)"
        elapsed=$((now - start_ts))
        if (( elapsed >= STARTUP_TIMEOUT )); then
            for camera in "${cameras[@]}"; do
                if [[ "${ready[$camera]:-0}" != "1" ]]; then
                    echo -e "${RED}Camera '$camera' failed to start within ${STARTUP_TIMEOUT}s.${NC}" >&2
                    echo -e "${YELLOW}Last output from '$camera':${NC}" >&2
                    capture_camera_output "$camera" 25 >&2
                    stop_session
                    exit 1
                fi
            done
        fi

        sleep 0.5
    done
}

cmd_start() {
    local replace=false
    local status_dir
    local arg camera encoder_gpu
    REQUESTED_CAMERAS=()

    while [[ $# -gt 0 ]]; do
        arg="$1"
        case "$arg" in
            -h|--help)
                usage
                exit 0
                ;;
            -c|--cameras)
                [[ $# -ge 2 ]] || die "--cameras requires a comma-separated value"
                append_camera_list "$2"
                shift 2
                ;;
            --startup-timeout)
                [[ $# -ge 2 ]] || die "--startup-timeout requires seconds"
                STARTUP_TIMEOUT="$2"
                [[ "$STARTUP_TIMEOUT" =~ ^[0-9]+$ ]] || die "--startup-timeout must be a positive integer"
                (( STARTUP_TIMEOUT > 0 )) || die "--startup-timeout must be greater than zero"
                shift 2
                ;;
            --replace)
                replace=true
                shift
                ;;
            *)
                append_camera_list "$arg"
                shift
                ;;
        esac
    done

    if [[ ${#REQUESTED_CAMERAS[@]} -eq 0 ]]; then
        REQUESTED_CAMERAS=("${DEFAULT_CAMERAS[@]}")
    fi

    require_tmux
    cd "$PROJECT_ROOT"

    if tmux_session_exists; then
        if [[ "$replace" == "true" ]]; then
            stop_session
        else
            die "tmux session '$SESSION_NAME' already exists. Use '$0 stop' or start with --replace."
        fi
    fi

    status_dir="${TMPDIR:-/tmp}/flash_aloha_camera_servers.${SESSION_NAME}"
    rm -rf "$status_dir"
    mkdir -p "$status_dir"

    echo -e "${BLUE}Starting camera servers in tmux session '$SESSION_NAME'...${NC}"
    echo "Cameras: ${REQUESTED_CAMERAS[*]}"
    if encoder_gpu="$(select_encoder_gpu)"; then
        echo "Encoder GPU: physical GPU $encoder_gpu (most free VRAM at startup)"
    else
        encoder_gpu=""
        echo -e "${YELLOW}Could not query GPU memory; leaving FFmpeg GPU selection automatic.${NC}"
    fi

    tmux new-session -d -s "$SESSION_NAME" -n manager -c "$PROJECT_ROOT" "sleep 86400"
    tmux set-option -t "$SESSION_NAME" history-limit 20000 >/dev/null

    for camera in "${REQUESTED_CAMERAS[@]}"; do
        start_camera_window "$camera" "$status_dir" "$encoder_gpu"
    done
    tmux kill-window -t "$SESSION_NAME:manager" >/dev/null 2>&1 || true

    wait_for_startup "$status_dir" "${REQUESTED_CAMERAS[@]}"

    echo
    echo -e "${GREEN}All requested camera servers are running.${NC}"
    echo "Attach: $0 attach"
    echo "Last 5 lines: $0 tail CAMERA"
    echo "Stop: $0 stop"
}

cmd_tail() {
    local camera="${1:-}"
    local lines="${2:-5}"
    [[ -n "$camera" ]] || die "tail requires a camera name"
    is_valid_camera "$camera" || die "unknown camera '$camera' (valid: ${VALID_CAMERAS[*]})"
    [[ "$lines" =~ ^[0-9]+$ ]] || die "lines must be a positive integer"

    require_tmux
    tmux_session_exists || die "tmux session '$SESSION_NAME' is not running"
    tmux_target_exists "$camera" || die "camera '$camera' is not running in session '$SESSION_NAME'"

    if [[ "$(tmux display-message -p -t "$SESSION_NAME:$camera" '#{pane_dead}')" == "1" ]]; then
        echo -e "${RED}Camera '$camera' window is not actively running.${NC}" >&2
        echo -e "${YELLOW}Last output from '$camera':${NC}" >&2
        capture_camera_output "$camera" "$lines" >&2
        exit 1
    fi

    capture_camera_output "$camera" "$lines"
}

cmd_status() {
    require_tmux
    tmux_session_exists || die "tmux session '$SESSION_NAME' is not running"
    tmux list-windows -t "$SESSION_NAME" -F '#{window_name}: #{?pane_dead,exited #{pane_dead_status},running}'
}

cmd_attach() {
    require_tmux
    tmux_session_exists || die "tmux session '$SESSION_NAME' is not running"
    tmux attach -t "$SESSION_NAME"
}

main() {
    local command="${1:-start}"
    [[ $# -gt 0 ]] && shift || true

    case "$command" in
        start)
            cmd_start "$@"
            ;;
        tail|logs|last)
            cmd_tail "$@"
            ;;
        status)
            cmd_status "$@"
            ;;
        attach)
            cmd_attach "$@"
            ;;
        stop)
            require_tmux
            stop_session
            echo "Stopped tmux session '$SESSION_NAME'."
            ;;
        __run-camera)
            run_camera "$@"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            die "unknown command '$command'"
            ;;
    esac
}

main "$@"
