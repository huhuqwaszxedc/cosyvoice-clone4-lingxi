#!/bin/bash

# 从文件中读取文本到数组
file_path="gen_text_list.txt"
tts_text_list=()
while IFS= read -r line; do
    tts_text_list+=("$line")
done < "$file_path"

# 设置 total_requests 为文件的行数
total_requests=$(wc -l < "$file_path")
completed_requests=0
running_processes=0
concurrent_requests=24

# Function: Start a new request
start_request() {
    if [ $completed_requests -lt $total_requests ]; then
        tts_text="${tts_text_list[$completed_requests]}"
        wav_file="result/gen_wavs/tts_009/009_prompt_v13_test_01/demo_$completed_requests.wav"
        python client.py --stream --tts_text "$tts_text" --output_path "$wav_file" &
        pid=$!
        ((running_processes++))
        ((completed_requests++))
        pids[$pid]=1
    fi
}

# Function: Wait for a certain number of processes to complete
wait_for_some_processes() {
    wait_count=$((RANDOM % 4 + 2))
    while [ $running_processes -gt $((concurrent_requests - wait_count)) ]; do
        for pid in "${!pids[@]}"; do
            if ! ps -p $pid > /dev/null; then
                unset pids[$pid]
                ((running_processes--))
            fi
        done
        sleep 0.01
    done
}

mkdir -p result/wavs

for ((i = 0; i < concurrent_requests; i++)); do
    start_request
done

# Continue requests until the total number of requests is reached
while [ $completed_requests -lt $total_requests ]; do
    wait_for_some_processes
    echo "concurrent_requests: $concurrent_requests running_processes: $running_processes"
    new_processes=$((concurrent_requests - running_processes))
    for ((i = 0; i < new_processes; i++)); do
        start_request
    done
done

wait
    