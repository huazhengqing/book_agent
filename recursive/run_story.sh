#!/bin/bash


MODEL=openai/deepseek-ai/DeepSeek-R1-0528
# MODEL=openrouter/deepseek/deepseek-r1-0528:free


task_input_file=../test/story.jsonl
output_folder=../test/story/
mkdir -p ${output_folder}
task_output_file=${output_folder}/output.jsonl
done_file=${output_folder}/done.txt


rm -rf ../test/story/爽文小说2/*
rm -rf ../recursive/.mem0
rm -rf ../recursive/run_story.log


rm -rf ../.litellm_cache
rm -rf ../.mem0
rm -rf ../test/log
rm -rf ../test/report


# docker-compose stop
docker-compose down -v


docker-compose up -d
docker ps -a | grep memgraph
docker ps -a | grep qdrant


source ../venv/bin/activate
python3 engine.py --filename $task_input_file --output-filename $task_output_file --done-flag-file $done_file --model ${MODEL} --mode story --language zh >> run_story.log 2>&1

