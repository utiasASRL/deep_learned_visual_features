#!/bin/bash
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet \
     --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
     'https://drive.google.com/uc?export=download&id=1xFGIJACmW6R_J2uzU6JAZKzQ6TBWDJ55' -O- | \
     sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xFGIJACmW6R_J2uzU6JAZKzQ6TBWDJ55" -O networks.zip \
     && rm -rf /tmp/cookies.txt
unzip networks.zip
rm networks.zip