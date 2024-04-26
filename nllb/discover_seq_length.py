import os

from transformers import AutoTokenizer

import sentencepiece as spm

from make_tokenizer import c2t

def nllb_tokenize(es: list, other: list, lens: list, lang_code: str):
    
    """
        Tokenizes the input sequences using the NLLB tokenizer.
        
        Parameters:
        - es (list): list of Spanish sentences
        - other (list): list of sentences in another language
        - lens (list): list of lengths of the tokenized sequences
        - lang_code (str): language code of the other language
    """
    
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/nllb-200-distilled-600M',
        src_lang='spa_Latn',
        tgt_lang=c2t[lang_code],
        use_fast=True,
        return_tensors='pt',
        padding=False,
        truncation=False
    )
    
    for i in range(len(es)):
        tokenized = tokenizer(
            text = es[i],
            text_target = other[i],
            return_tensors = 'pt',
            padding=False,
            truncation=False
        )
        input_ids = tokenized['input_ids'][0]
        labels = tokenized['labels'][0]
        lens.extend([len(input_ids), len(labels)])
        
def mamba_tokenize(es: list, other: list, lens: list, tokenizer_type: str):
    
    """
        Tokenizes the input sequences using the SentencePiece tokenizer.
        
        Parameters:
        - es (list): list of Spanish sentences
        - other (list): list of sentences in another language
        - lens (list): list of lengths of the tokenized sequences
        - tokenizer_type (str): type of the SentencePiece tokenizer
    """
    
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(os.path.join('vocab', tokenizer_type + '.model'))
    
    for i in range(len(es)):
        tokenized = tokenizer.EncodeAsPieces(es[i] + other[i])
        lens.append(len(tokenized) + 1) # +1 for the <lang> token

def main():
    
    #tokenizer_type = 'nllb' # 756 max len
    tokenizer_type = 'unigram' # 775 max len
    #tokenizer_type = 'bpe' # 772 max len

    dev_dir = os.path.join('proj_data_final', 'dev')

    lens = []

    for file in os.listdir(dev_dir):

        with open(os.path.join(dev_dir, file), 'r', encoding='utf-8') as f:
            
            lang_code = file.split('.')[0]
            
            es = []
            other = []
            
            lines = f.readlines()
            for line in lines:
                es_line, other_line = line.split('\t')
                es.append(es_line.strip())
                other.append(other_line.strip())

            if tokenizer_type == 'nllb':
                nllb_tokenize(es, other, lens, lang_code)
            else:
                mamba_tokenize(es, other, lens, tokenizer_type)

    print(sorted(lens, reverse=True))
    print(f'Used tokenizer: {tokenizer_type}, max length: {max(lens)}')
    
if __name__ == '__main__':
    main()