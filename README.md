## Overview

This is the test code for the Textured Neural Avatar project (https://saic-violet.github.io/texturedavatar/).
We provide two avatar models for the datasets available from Alldieck et al. https://graphics.tu-bs.de/people-snapshot (the original datasets should be requested from their authors). Our models were trained on two videos (female-1-casual, male-2-casual) and can generate new views for new OpenPose-format poses (some test poses provided).

## Instructions

Build docker image:

```
$ docker build . -t textured_avatars
```

Run container:

```
$ bash run.sh
```

Build stickman drawer:

```
$ bash build_stickman.sh
```

Render images (female-1-casual):

```
$ python test.py --model person_1
```

Render images (male-2-casual):

```
$ python test.py --model person_2
```


