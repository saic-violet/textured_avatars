#!/bin/bash

docker run -it --rm \
    -v $(pwd):/src \
    -w /src \
    textured_avatars \
    bash