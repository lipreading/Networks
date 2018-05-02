#!/usr/bin/env bash

SOURCE_DIR='test_frames/aaa'  # папка с изначальными кадрами
DESTINATION_DIR='test_frames/destination'  # папка, куда надо переместить тестовую выборку

echo 'Start moving words for test dataset from '${SOURCE_DIR}' to '${DESTINATION_DIR};
i=0;
for F in ${SOURCE_DIR}/*; do
echo 'i='${i}
if [ ${i} -eq $((i/10*10)) ];
then echo ${F};
#then mv ${F} ${DESTINATION_DIR};
fi;
let i+=1
done
echo 'Finished creating test dataset'
