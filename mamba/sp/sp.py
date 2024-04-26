import os

import sentencepiece as spm

input_for_training = os.path.join('vocab.txt') # contains all text to train vocab on
model_name = os.path.join('unigram_16k')
train = True
#train = False # training is done already

"""TRAINING A SENTENCEPIECE TOKENIZER BELOW"""

if train: # fold up if statement to see tests/usage below

    params = ' --input=' + input_for_training
    
    params += ' --model_prefix=' + model_name

    params += ' --vocab_size=16000'

    params += ' --character_coverage=1.0'

    params += ' --normalization_rule_name=nfkc'

    #params = params + ' --model_type=unigram'
    params += ' --model_type=unigram'
    
    # params += ' --user_defined_symbols=<pad>'

    params += ' --control_symbols=' + ','.join(
        [
            '<pad>',
            '<aym>',
            '<bzd>',
            '<cni>',
            '<ctp>',
            '<gn>',
            '<hch>',
            '<nah>',
            '<oto>',
            '<quy>',
            '<shp>',
            '<tar>'
        ]
    )

    params += ' --shrinking_factor=0.95' # limit
    
    spm.SentencePieceTrainer.Train(params)
 
"""TEST A MODEL BELOW AFTER IT IS TRAINED"""

sp = spm.SentencePieceProcessor()
sp.Load(model_name + '.model')

print(sp.__dict__)
print(sp.this)

print(sp.EncodeAsPieces('Hola mundo.'))
print(sp.EncodeAsIds('Hola mundo.'))

print()
print(sp.EncodeAsPieces('hoy tenemos que entrar a la ciudad porque hay un evento.'))
print(sp.EncodeAsIds('hoy tenemos que entrar a la ciudad porque hay un evento.'))

print()
print(sp.DecodeIds([10, 30, 60, 100, 1000]))

print()
print('<pad>', sp.PieceToId('<pad>'))
print('<shp>', sp.PieceToId('<shp>'))
print('<tar>', sp.PieceToId('<tar>'))
print(
    sp.DecodeIds(
        sp.EncodeAsIds(
            "Él se quedó con ellos en Nueva York."
        ) + [sp.PieceToId('<bzd>')] + sp.EncodeAsIds(
            "Ie' ẽ' tsèxãt i yàmĩ tã Nueva York."
        )
    )
)