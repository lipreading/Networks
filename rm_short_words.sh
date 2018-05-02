#!/usr/bin/env bash

FRAME_DIR='test_frames'  # работает для ВСЕГО датасета, то есть это папка с разными видео

let border=4  # граница отсева коротких слов

echo 'Start removing short words from dataset'
for word in ${FRAME_DIR}/*; do
#    for word in ${video}/*; do
        let LENGTH=`python3 'rm_short_words.py' ${word}`;  # присваиваем переменной результат выполнения python скрипта (длину слова)
        if ((${LENGTH} < ${border}));
            then rm -r ${word}; echo ${word};
        fi;
#    done;
done;
echo 'Removing finished'
