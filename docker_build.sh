#!/bin/bash
if [ -z $1 ]
then
   echo "s1:version"
   exit
fi

if [ -z $DOCKER_REGISTRY ]
then
   echo "not find $DOCKER_REGISTRY"
   exit
fi

VERSION=$1

##1.create docker regsit
docker build -t $DOCKER_REGISTRY/tianchi-submit:$VERSION .

##2.PUSH
#docker push $DOCKER_REGISTRY/tianchi-submit:$VERSION

