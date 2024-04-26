import os

dirs = [
    'bad_supp',
    'dev',
    'good_supp',
    'train'
]

for dir in dirs:
    files = [os.path.join('data', dir, f) for f in os.listdir(dir)]
    for file in files:
        lines = open(file, 'r', encoding='utf-8').readlines()
        for i in range(len(lines)):
            if len(lines[i].split('\t')) != 2:
                print(f'File: {file}\nLine: {i}')
