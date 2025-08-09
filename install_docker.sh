#!/bin/bash


sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://3xoj0j3i.mirror.aliyuncs.com",
    "https://docker.m.daocloud.io",
    "https://mirror.azure.cn",
    "https://ghcr.hub1.nat.tf",
    "https://f1361db2.m.daocloud.io"
  ]
}
EOF
systemctl daemon-reload
systemctl restart docker
docker info | grep "Registry Mirrors" -A 5

