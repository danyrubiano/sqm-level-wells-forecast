#!/bin/bash
_CONTAINER_REGISTRY=us-central1-docker.pkg.dev
_PROJECT_ID=levels-sqm
_TEAM_NAME=rj
_PROJECT_NAME=forecast-level-wells-sqm
_IMG_NAME=preprocess
_RELEASE=latest

# docker build --tag $img_name:$version .
docker build --tag $_CONTAINER_REGISTRY/$_PROJECT_ID/$_TEAM_NAME/$_PROJECT_NAME/$_IMG_NAME:$_RELEASE .
docker push $_CONTAINER_REGISTRY/$_PROJECT_ID/$_TEAM_NAME/$_PROJECT_NAME/$_IMG_NAME:$_RELEASE