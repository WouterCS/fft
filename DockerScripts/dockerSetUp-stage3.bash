NAME='tfGPU'
IMAGENAME='myimage2' # 'mydockerimage' #gcr.io/tensorflow/tensorflow:latest-gpu /bin/bash
DROPBOXPATH='/home/uijenswr/Dropbox/thesis'


PROJECTPATH="$(dirname "$(dirname "${BASH_SOURCE[0]}")")" #'/home/wouter/Documents/DockerMap/Code'
docker rm $(docker stop $(docker ps -aq --no-trunc))
nvidia-docker run -itd -p 8888:8888 -p 6006:6006 --name $NAME -v $PROJECTPATH/notebooks:/notebooks -v $PROJECTPATH/RFNN:/usr/local/lib/python2.7/dist-packages/RFNN $IMAGENAME 

cd $PROJECTPATH
git pull
docker cp $PROJECTPATH/RFNN     $NAME:/RFNN
docker cp $PROJECTPATH/DockerScripts/DockerCreation $NAME:/scripts
docker start $NAME 
docker exec -it $NAME /bin/bash /scripts/ondockerSetUp.sh
docker cp $NAME:/results                   $DROPBOXPATH
docker exec -it $NAME /bin/bash
