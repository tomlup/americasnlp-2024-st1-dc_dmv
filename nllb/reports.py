import os

reports_dir = os.path.join('outputs', 'reports')

ckpts = [os.path.join(reports_dir, ckpt) for ckpt in os.listdir(reports_dir)]

best_scores = {
    'aym': 0.0,
    'bzd': 0.0,
    'cni': 0.0,
    'ctp': 0.0,
    'gn':  0.0,
    'hch': 0.0,
    'nah': 0.0,
    'oto': 0.0,
    'quy': 0.0,
    'shp': 0.0,
    'tar': 0.0
}

best_ckpts = {
    'aym': '',
    'bzd': '',
    'cni': '',
    'ctp': '',
    'gn':  '',
    'hch': '',
    'nah': '',
    'oto': '',
    'quy': '',
    'shp': '',
    'tar': ''
}

for ckpt in ckpts:
    
    good = True
    
    langs = []
    scores = []
    
    for file in os.listdir(ckpt):        
        file_path = os.path.join(ckpt, file)
        
        lang = os.path.basename(file_path)[:-4]
        
        lines = open(file_path, 'r', encoding='utf-8').readlines()
        
        try:
            if 'assert' in lines[3]:
                print('unusable', os.path.basename(ckpt))
                good = False
                break
        except:
            print(file_path)
            exit()
            
        score = float(lines[3].split()[-1])
        
        langs.append(lang)
        scores.append(score)
        
    if len(langs) != 11 or len(scores) != 11:
        good = False
    
    for lang, score in zip(langs, scores):
        if score > best_scores[lang]:
            best_scores[lang] = score
            best_ckpts[lang] = os.path.basename(ckpt)

    for score in scores:
        if score < 12.1:
            good = False
            break
    
    if good:
        print()
        print('good', os.path.basename(ckpt))
        for i in range(11):
            print(f'{langs[i]}: {scores[i]}')
        print()
    
for lang_code in best_ckpts:
    print()
    print(f'{lang_code}: {best_ckpts[lang_code]}')
    print(f'{lang_code}: {best_scores[lang_code]}')

print('\nto submit:')
print(set(best_ckpts.values()))