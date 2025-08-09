#!/bin/bash


docker pull memgraph/memgraph-mage:latest
docker pull memgraph/lab:latest


docker volume create memgraph-data


docker run -d -p 7687:7687 -p 7444:7444 -e MGCONSOLE="--username memgraph --password memgraph" -v memgraph-data:/var/lib/memgraph --name memgraph memgraph/memgraph-mage:latest --schema-info-enabled=True


docker run -d -p 3000:3000 --name memgraph_lab memgraph/lab:latest


docker ps -a | grep memgraph


docker stop -t 60   memgraph
docker restart memgraph


create user memgraph identified by 'memgraph'



docker-compose up -d























