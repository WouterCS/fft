NAME='tfGPU'
IMAGENAME='myimage1' # 'mydockerimage' #gcr.io/tensorflow/tensorflow:latest-gpu /bin/bash
PROJECTPATH="$(dirname "$(dirname "${BASH_SOURCE[0]}")")" #'/home/wouter/Documents/DockerMap/Code'

#cd $PROJECTPATH/DockerCreation/
#docker build --tag $IMAGENAME ./

nvidia-docker run -itd -p 8888:8888 -p 6006:6006 --name $NAME -v $PROJECTPATH/notebooks:/notebooks -v $PROJECTPATH/RFNN:/usr/local/lib/python2.7/dist-packages/RFNN $IMAGENAME 

docker cp $PROJECTPATH/nn.py      $NAME:/usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/nn.py
docker cp $PROJECTPATH/user_ops $NAME:/user_ops
docker cp $PROJECTPATH/RFNN     $NAME:/RFNN
docker cp $PROJECTPATH/DockerScripts/DockerCreation $NAME:/scripts
docker start $NAME 
docker exec -it $NAME /bin/bash /scripts/ondockerSetUp.sh
docker cp $NAME:/results                   $PROJECTPATH
docker exec -it $NAME /bin/bash
