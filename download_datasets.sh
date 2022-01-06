#!/bin/bash
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet \
     --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
     'https://drive.google.com/uc?export=download&id=10yOGwbGDhJEhbm0h4faiy-lO-7NPGJEQ' -O- | \
     sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10yOGwbGDhJEhbm0h4faiy-lO-7NPGJEQ" -O datasets.zip \
     && rm -rf /tmp/cookies.txt
unzip datasets.zip
rm datasets.zip