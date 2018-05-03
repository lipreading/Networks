#!/usr/bin/env bash

FRAME_DIR='test_frames'  # работает для ВСЕГО датасета, то есть это папка с разными видео

let border=5  # граница отсева слов с маленьким кол-вом кадров

echo 'Start removing words with few frames from dataset'
for video in ${FRAME_DIR}/*; do
    for word in ${video}/*; do
        let LENGTH=`python3 'rm_few_frames.py' ${word}`;  # присваиваем переменной результат выполнения python скрипта (длину слова)
        if ((${LENGTH} < ${border}));
            then rm -r ${word}; echo ${word};
        fi;
    done;
done;
echo 'Removing finished'
