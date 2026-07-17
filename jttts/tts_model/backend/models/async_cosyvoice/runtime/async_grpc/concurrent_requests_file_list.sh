#!/bin/bash

tts_text_list=("欲穷千里目，更上一层楼。" "白日依山尽，黄河入海流。" "明月松间照，清泉石上流。" "竹喧归浣女，莲动下渔舟。" "飞流直下三千尺，疑是银河落九天。" "蛾儿雪柳黄金缕，笑语盈盈暗香去。" "看到小猫咪的样子，笑着说，小猫咪，你变成这样也好可爱呀。" "欲穷千里目，更上一层楼。" "白日依山尽，黄河入海流。" "明月松间照，清泉石上流。" "竹喧归浣女，莲动下渔舟。" "飞流直下三千尺，疑是银河落九天。" "蛾儿雪柳黄金缕，笑语盈盈暗香去。" "看到小猫咪的样子，笑着说，小猫咪，你变成这样也好可爱呀。" "欲穷千里目，更上一层楼。" "白日依山尽，黄河入海流。" "明月松间照，清泉石上流。" "竹喧归浣女，莲动下渔舟。" "飞流直下三千尺，疑是银河落九天。" "蛾儿雪柳黄金缕，笑语盈盈暗香去。" "看到小猫咪的样子，笑着说，小猫咪，你变成这样也好可爱呀。" "欲穷千里目，更上一层楼。" "白日依山尽，黄河入海流。" "明月松间照，清泉石上流。" "竹喧归浣女，莲动下渔舟。" "飞流直下三千尺，疑是银河落九天。" "蛾儿雪柳黄金缕，笑语盈盈暗香去。" "看到小猫咪的样子，笑着说，小猫咪，你变成这样也好可爱呀。")

tts_text_list_length=${#tts_text_list[@]}
echo "tts_text_list_length: $tts_text_list_length"
total_requests=$tts_text_list_length
completed_requests=0
running_processes=0
concurrent_requests=24

# Function: Start a new request
start_request() {
    if [ $completed_requests -lt $total_requests ]; then
        tts_text="${tts_text_list[$completed_requests]}"
        wav_file="result/gen_wavs/tts_009/009_prompt_v13_test_debug/demo_$completed_requests.wav"
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
    