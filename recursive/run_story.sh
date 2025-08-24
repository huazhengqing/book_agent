#!/bin/bash


task_input_file=../test/story.jsonl
output_folder=../test/story/
mkdir -p ${output_folder}
task_output_file=${output_folder}/output.jsonl
done_file=${output_folder}/done.txt



rm -rf ./.mem0
rm -rf ./.cache
rm -rf ./run_story.log


rm -rf ../.litellm_cache
rm -rf ../.mem0
rm -rf ../.cache
rm -rf ../test/log
rm -rf ../test/report
rm -rf ../test/story/爽文小说2/*


# docker-compose stop
# docker-compose down -v
# docker-compose up -d
# sleep 10
docker ps -a | grep memgraph
docker ps -a | grep qdrant


source ../venv/bin/activate
python3 engine.py --filename $task_input_file --output-filename $task_output_file --done-flag-file $done_file --mode story --language zh >> run_story.log 2>&1

