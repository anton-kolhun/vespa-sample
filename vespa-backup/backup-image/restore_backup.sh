#!/bin/bash

if [ $# -ne 1 ]
  then
    echo "BACKUP_DATE argument should be supplied: ./restore_backup.sh 2026/8/21"
    exit -1
fi

BACKUP_DATE=$1
S3_PATH="$VESPA_BACKUP_S3_BUCKET"/"$BACKUP_DATE"

NUMBER_OF_SLICES=$(aws s3 ls "$S3_PATH"/ | grep backup_[[:digit:]] | wc -l)

set +e

for ((i = 0 ; i < NUMBER_OF_SLICES ; i++));
    do aws s3 cp "$S3_PATH"/backup_"$i".json.gz .
    gzip -f -d backup_"$i".json.gz
    vespa-feeder backup_"$i".json --verbose --abortondataerror false --abortonsenderror false > backup_"$i".log 2>&1
    rm -f backup_"$i".json
done
