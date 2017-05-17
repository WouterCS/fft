NAME='tfGPU'
IMAGENAME='myimage1' # 'mydockerimage' #
PROJECTPATH="$(dirname "$(dirname "${BASH_SOURCE[0]}")")" #'/home/wouter/Documents/DockerMap/Code'
git pull
docker rm $(docker stop $(docker ps -aq --no-trunc))
nvidia-docker run -itd -p 8888:8888 -p 6006:6006 --name $NAME -v $PROJECTPATH/notebooks:/notebooks -v $PROJECTPATH/RFNN:/usr/local/lib/python2.7/dist-packages/RFNN $IMAGENAME 

docker cp $PROJECTPATH/nn.py      $NAME:/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn.py
docker cp $PROJECTPATH/user_ops $NAME:/user_ops
docker cp $PROJECTPATH/DockerScripts/DockerCreation $NAME:/scripts
docker start $NAME 
docker exec -it $NAME /bin/bash /scripts/ondockerSetUp-stage2.bash

docker commit $NAME 'myimage2'
bash $PROJECTPATH/DockerScripts/dockerSetUp-stage3.bash