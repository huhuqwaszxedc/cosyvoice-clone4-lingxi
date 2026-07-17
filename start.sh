#!/usr/bin/env bash

# ==============================
# TTS 服务启动与智能监控脚本（支持长启动 + 两级健康检查）
# ==============================

# set -euo pipefail

# --- 配置区 ---
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "/tts_deploy_info/conda_envs/clone4.0" ]; then
    SERVER_LOG_DIR="/tts_deploy_info/server_log"
else
    SERVER_LOG_DIR="$WORKSPACE_DIR/../server_log"
fi

# 获取容器/Pod 名称（使用 HOSTNAME 环境变量）
CONTAINER_NAME="${HOSTNAME:-unknown}"

MONITOR_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
mkdir -p "$SERVER_LOG_DIR"
LOG_FILE="$SERVER_LOG_DIR/${CONTAINER_NAME}_monitor_${MONITOR_TIMESTAMP}.log"

PID_FILE="/tmp/tts_service.pid"
HEALTH_CHECK_URL="http://localhost:7820/jttts/TTSInfer"
PORT=7820

# ===== 启动阶段配置 =====
STARTUP_CHECK_INTERVAL=60    # 启动期每次检查间隔（秒）→ 5分钟
STARTUP_MAX_RETRIES=70         # 最大重试次数 → 总超时 ≈ 35分钟

# ===== 监控阶段配置（两级策略）=====
NORMAL_CHECK_INTERVAL=180     # 正常运行时检查间隔（秒）→ 3分钟
FAILURE_CHECK_INTERVAL=60     # 首次失败后检查间隔（秒）→ 60秒
MAX_FAILURES_BEFORE_RESTART=3 # 连续失败多少次后重启


# ===== 其他配置 =====
CURL_TIMEOUT=60               # 单次健康检查最大耗时（秒）

# ==============================

# --- 日志函数 ---
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg" >&2
    echo "$msg" >> "$LOG_FILE"
}

# --- 启动原始服务 ---
start_original_service() {
    cd "$WORKSPACE_DIR"

    if [ -d "/tts_deploy_info/conda_envs/clone4.0" ]; then
        PYTHON_EXEC="/tts_deploy_info/conda_envs/clone4.0/bin/python"
        log "Using Conda environment: /tts_deploy_info/conda_envs/clone4.0"
    elif [ -d "/root/miniconda3/envs/clone4.0" ]; then
        PYTHON_EXEC="/root/miniconda3/envs/clone4.0/bin/python"
        log "Using Conda environment: /root/miniconda3/envs/clone4.0"
    else
        log "[Error]: clone4.0 Conda environment not found!"
        exit 1
    fi

    if [ ! -x "$PYTHON_EXEC" ]; then
        log "[Error]: Python executable not found: $PYTHON_EXEC"
        exit 1
    fi

    if [ -f "main.pyc" ]; then
        MAIN_SCRIPT="main.pyc"
    else
        MAIN_SCRIPT="main.py"
    fi

    SERVER_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    LOG_PATH="$SERVER_LOG_DIR/${CONTAINER_NAME}_service_${SERVER_TIMESTAMP}.log"
    log "Starting service (may take up to 30 minutes), log: $LOG_PATH"

    "$PYTHON_EXEC" "${MAIN_SCRIPT}" > "$LOG_PATH" 2>&1 &
    echo $!
}

# --- 健康检查（关键：绕过代理 + 超时）---
health_check() {
    log "Performing health check: curl $HEALTH_CHECK_URL"

    # curl http://localhost:7820/jttts/TTSInfer -H "Content-Type:application/json" -X POST -d '{ "text": "服务监控", "user_id": "jinya" }'

    local json_data='{ "text": "服务监控", "user_id": "jinya" }'

    # Clear proxy environment variables to force direct connection
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

    local status_code
    status_code=$(curl -s -w "%{http_code}" \
        -o /dev/null \
        --noproxy "*" \
        --connect-timeout 10 \
        --max-time "$CURL_TIMEOUT" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "$json_data" \
        "$HEALTH_CHECK_URL")

    if [[ "$status_code" == "200" ]]; then
        log "[success] Health check passed (HTTP $status_code)"
        return 0
    else
        log "[failed] Health check failed (HTTP $status_code)"
        return 1
    fi
}

kill_python_process(){
    # 设置要杀死的进程名称列表
    PROCESS_NAMES=(
        "clone4.0/bin/python"
        "main.py"
        "main.pyc"
        "register_service.pyc"
        "register_service.py"
    )

    # 循环执行10次
    for ((i=1; i<=10; i++)); do
        log "Round $i started..."

        # 遍历进程名称列表
        for process_name in "${PROCESS_NAMES[@]}"; do
            log "Searching and killing processes: $process_name"

            # 获取进程ID列表
            pids=$(ps aux | grep "$process_name" | grep -v grep | awk '{print $2}')

            if [ -n "$pids" ]; then
                log "Found PIDs: $pids"

                # 先尝试优雅终止
                log "Sending SIGTERM signal..."
                kill -15 $pids 2>/dev/null

                # 等待2秒让进程有时间退出
                sleep 2

                # 检查进程是否还在运行
                remaining_pids=$(ps aux | grep "$process_name" | grep -v grep | awk '{print $2}')

                if [ -n "$remaining_pids" ]; then
                    log "Process still running, sending SIGKILL signal..."
                    kill -9 $remaining_pids 2>/dev/null
                    log "Process forcefully terminated"
                else
                    log "Process exited successfully"
                fi
            else
                log "No matching process found"
            fi

            log "------------------------"
        done

        # 等待100毫秒
        sleep 0.1
    done

    log "kill python execution completed"
}


# --- 终止旧服务（含 GPU 进程清理）---
cleanup_old_service() {
    # Terminate main process
    local old_pid=""
    if [[ -f "$PID_FILE" ]]; then
        old_pid=$(grep -E '^[0-9]+$' "$PID_FILE" | head -n1)
        if [[ -n "$old_pid" && "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
            log "Terminating old service main process PID=$old_pid"
            kill "$old_pid" 2>/dev/null || true
            wait "$old_pid" 2>/dev/null || true
        fi
        rm -f "$PID_FILE"
    fi

    # --- Safe GPU process cleanup (only if nvidia-smi is available) ---
    if command -v nvidia-smi >/dev/null 2>&1; then
        log "Cleaning up GPU processes..."
        gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null) || gpu_pids=""
        if [[ -n "$gpu_pids" ]]; then
            echo "$gpu_pids" | while read -r pid; do
                if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
                    log "Terminating GPU process PID=$pid"
                    kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
                fi
            done
        else
            log "No GPU processes detected"
        fi
    else
        log "nvidia-smi not available, skipping GPU cleanup"
    fi

    kill_python_process

}

# --- 启动主服务（等待就绪）---
launch_service() {
    log "Starting TTS service (timeout: $((STARTUP_CHECK_INTERVAL * STARTUP_MAX_RETRIES / 60)) minutes)..."
    cleanup_old_service

    SERVICE_PID=$(start_original_service)
    printf '%d\n' "$SERVICE_PID" > "$PID_FILE"
    log "Service started with PID=$SERVICE_PID, waiting for readiness..."

    for i in $(seq 1 "$STARTUP_MAX_RETRIES"); do
        log "Health check attempt $i (waited $(((i-1) * STARTUP_CHECK_INTERVAL / 60)) minutes)..."
        if health_check; then
            log "[success] Service is ready!"
            return 0
        fi
        if [ $i -lt "$STARTUP_MAX_RETRIES" ]; then
            sleep "$STARTUP_CHECK_INTERVAL"
        fi
    done

    log "Warning: Service not ready after $((STARTUP_CHECK_INTERVAL * STARTUP_MAX_RETRIES / 60)) minutes"
}

# --- 主流程（两级健康检查状态机）---
main() {
    log "=== TTS Service Monitor Started ==="
    log "Startup timeout: $((STARTUP_CHECK_INTERVAL * STARTUP_MAX_RETRIES / 60)) minutes"
    log "Normal check interval: ${NORMAL_CHECK_INTERVAL} seconds"
    log "Failure check interval: ${FAILURE_CHECK_INTERVAL} seconds"
    log "Restart after ${MAX_FAILURES_BEFORE_RESTART} consecutive failures"

    launch_service

    # Initial health check
    if ! health_check; then
        log "Initial health check failed, attempting restart..."
        launch_service
        if ! health_check; then
            log "Still failing after restart, entering monitoring loop"
        fi
    fi

    # ===== Two-stage health check logic =====
    local consecutive_failures=0
    local next_sleep="$NORMAL_CHECK_INTERVAL"

    while true; do
        sleep "$next_sleep"

        if health_check; then
            if [ $consecutive_failures -gt 0 ]; then
                log "[success] Health check recovered! Back to normal mode"
            fi
            consecutive_failures=0
            next_sleep="$NORMAL_CHECK_INTERVAL"
        else
            ((consecutive_failures++))
            log "Consecutive failures: $consecutive_failures / $MAX_FAILURES_BEFORE_RESTART"

            if [ $consecutive_failures -ge "$MAX_FAILURES_BEFORE_RESTART" ]; then
                log "[Warning] Max consecutive failures reached, restarting service..."
                launch_service
                consecutive_failures=0
                next_sleep="$NORMAL_CHECK_INTERVAL"
            else
                next_sleep="$FAILURE_CHECK_INTERVAL"
            fi
        fi
    done
}

# --- 优雅退出处理 ---
cleanup() {
    log "Received termination signal, shutting down service..."
    if [[ -f "$PID_FILE" ]]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

# --- 执行主流程 ---
main