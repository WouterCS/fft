NAME='tfGPU'
IMAGENAME='tensorflow/tensorflow:nightly-devel-gpu /bin/bash' # 'mydockerimage' #
PROJECTPATH="$(dirname "$(dirname "${BASH_SOURCE[0]}")")" #'/home/wouter/Documents/DockerMap/Code'


docker rm $(docker stop $(docker ps -aq --no-trunc))
nvidia-docker run -itd -p 8888:8888 -p 6006:6006 --name $NAME -v $PROJECTPATH/notebooks:/notebooks -v $PROJECTPATH/RFNN:/usr/local/lib/python2.7/dist-packages/RFNN $IMAGENAME 

docker cp $PROJECTPATH/DockerScripts/DockerCreation $NAME:/scripts
docker start $NAME 
docker exec -it $NAME /bin/bash /scripts/ondockerSetUp-stage1.bash

docker commit $NAME 'myimage1'
bash $PROJECTPATH/DockerScripts/dockerSetUp-stage2.bash