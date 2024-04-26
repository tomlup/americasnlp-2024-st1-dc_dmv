import os

# README: set to True to check dev translations, False to check test translations
use_dev = True

translations = os.path.join('outputs', 'translations')
if not use_dev:
    translations += '_test'

ckpts = [os.path.join(translations, ckpt) for ckpt in os.listdir(translations)]

bad_ckpts = set()

for ckpt in ckpts:
    
    files = [os.path.join(ckpt, f) for f in os.listdir(ckpt)]
    
    # print()
    # print(os.path.basename(ckpt))
    
    for file in files:
        
        # print(os.path.basename(file))
        
        lines = open(file, 'r', encoding='utf-8').readlines()
        
        try:
            if not use_dev:
                if file.endswith('ctp.txt'):
                    assert len(lines) == 1000
                    continue
                assert len(lines) == 1003
            else:
                if file.endswith('tar.txt') or file.endswith('gn.txt'):
                    assert len(lines) == 995
                    continue
                if file.endswith('cni.txt'):
                    assert len(lines) == 883
                    continue
                if file.endswith('ctp.txt'):
                    assert len(lines) == 499
                    continue
                if file.endswith('oto.txt'):
                    assert len(lines) == 599
                    continue
                if file.endswith('nah.txt'):
                    assert len(lines) == 672
                    continue
                if file.endswith('hch.txt'):
                    assert len(lines) == 994
                    continue
                assert len(lines) == 996
        except AssertionError as e:
            # print(e)
            # print(f'{os.path.basename(ckpt)} failed on {os.path.basename(file)[:-4]}: {len(lines)}.')
            bad_ckpts.add(os.path.basename(ckpt))

print()
for bad_ckpt in bad_ckpts:
    print(f'Bad checkpoint: {bad_ckpt}')
    print()