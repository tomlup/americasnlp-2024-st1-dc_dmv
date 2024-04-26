import os
import subprocess

use_test = False

def execute_command(command, output_file):
    with open(output_file, 'w+', encoding='utf-8') as outfile:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            outfile.write(line)
        process.wait()
    print(f'Output written to {output_file}.')

tr_dir = os.path.join('outputs', 'translations')
if use_test:
    tr_dir += '_test'
ckpts = [os.path.join(tr_dir, ckpt) for ckpt in os.listdir(tr_dir) if os.path.isdir(os.path.join(tr_dir, ckpt))]

reports_dir = os.path.join('outputs', 'reports')
if use_test:
    reports_dir += '_test'
if not os.path.exists(reports_dir):
    os.mkdir(reports_dir)

for ckpt in ckpts:
    
    ckpt = os.path.basename(ckpt)
    
    # if not ckpt.startswith('checkpoint10_train_False'):
    #     continue
    
    # print(ckpt)
    
    if not os.path.exists(os.path.join(reports_dir, ckpt)):
        os.mkdir(os.path.join(reports_dir, ckpt))
    
    for tr_file in os.listdir(os.path.join(tr_dir, ckpt)):
        tr_file_path = os.path.join(tr_dir, ckpt, tr_file)
        lang_code = tr_file.split('.')[0]
        gold_dir = 'ref' if not use_test else 'test'
        gold_file_path = os.path.join('proj_data_final', gold_dir, lang_code+'.txt')
        
        command = [
            'python',
            'evaluate.py',
            '--sys',
            tr_file_path,
            '--ref',
            gold_file_path,
            '--detailed_output'
        ]
        
        output_file = os.path.join(reports_dir, ckpt, lang_code+'.txt')
        
        execute_command(command, output_file)