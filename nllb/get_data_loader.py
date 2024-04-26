from os import path, listdir

from typing import Union

from random import seed, choice, choices

from torch import manual_seed
from torch.utils.data import DataLoader, Dataset

from transformers.tokenization_utils_base import BatchEncoding

from make_tokenizer import make_tokenizer, c2t

seed(42) # for reproducibility
manual_seed(42)

def collate_fn(batch):
    
    """
        Removes the extra dimension that DataLoader adds.
        
        Parameters:
        - batch (list[transformers.tokenization_utils_base.BatchEncoding]): The batch to process.
        
        Returns:
        - transformers.tokenization_utils_base.BatchEncoding: The batch with the extra dimension removed.
    """
    
    return batch[0]

class TrainDataset(Dataset):
    
    """
        This dataset encapsulates the training data.
        
        Parameters:
        - split (str): The name of the split ('train' or 'good_supp' or 'bad_supp').
        - files (list[str]): A list of file paths to the training data.
        - batch_size (int): The number of examples in each batch.
        - num_batches (int): The number of batches to load.
    """
    
    def __init__(
        self,
        split: str,
        files: list[str],
        batch_size: int,
        num_batches: int,
        get_tokenized: bool,
        max_length: int
    ):
        
        super().__init__()
        
        # tokenize during getitem or not
        self.get_tokenized = get_tokenized
        if self.get_tokenized:
            self.tokenizers = dict.fromkeys(c2t.values())
            self.max_length = max_length
            for lang_token in self.tokenizers:
                self.tokenizers[lang_token] = make_tokenizer(lang_token, 'spa_Latn', self.max_length)
        
        # list of tokenized batches, each batch is the same target language
        # but different batches may differ in target language
        # since they are pre-tokenized, they are ready to be used in training without worrying
        # about which tokenizer to use
        self.examples = []
        
        # lists to be sampled from later to build self.examples
        temp = dict.fromkeys(c2t.values())
        temp.pop('spa_Latn', None)
        for lang_token in temp:
            temp[lang_token] = []
        
        # process each file (language) and add list of examples to temp
        for file in files:

            # tokenizer from spanish to this language
            lang_token = c2t[path.basename(file).split('.')[0]]
            # tokenizer = tokenizers[lang_token]
            # assert tokenizer._src_lang == 'spa_Latn'
            # assert tokenizer.tgt_lang == lang_token
            
            # lists of spanish and other language sentences
            es_batch = []
            other_batch = []
            
            lines = open(file, 'r', encoding='utf-8').readlines()
            for i in range(len(lines)):
                
                strip_line = lines[i].strip()
                if not strip_line:
                    continue
                split_line = strip_line.split('\t')
                if len(split_line) != 2:
                    continue
                
                # accumulate a batch of es sentences and other language sentences
                es_batch.append(split_line[0].strip())
                other_batch.append(split_line[1].strip())
                
                if len(es_batch) == batch_size or i == len(lines) - 1:
                    
                    assert len(es_batch) == len(other_batch)
                    
                    # tokenize a batch and append to temp dict
                    # tokenized = tokenizers[lang_token](
                    #     text = es_batch,
                    #     text_target = other_batch,
                    #     return_tensors = 'pt',
                    #     padding = 'longest',
                    #     truncation = True,
                    #     max_length = max_length
                    # )
                    # temp[lang_token].append(tokenized)
                    temp[lang_token].append((es_batch, other_batch, lang_token))
                    
                    # clear batch accumulation
                    es_batch = []
                    other_batch = []
                    
                    # no way we could need this much data
                    if i >= num_batches * batch_size:
                        break
                    
            print(f'Loaded {len(temp[lang_token]) * batch_size}/{len(lines)} lines of {lang_token}.')

        # build pmf using exponential reweighting
        probs = dict.fromkeys(c2t.values())
        probs.pop('spa_Latn', None)
        for lang_token in probs:
            probs[lang_token] = 0
            try:
                probs[lang_token] = len(temp[lang_token])**(-0.4)
            except ZeroDivisionError:
                pass
        total = sum(probs.values())
        probs = {lang_token: probs[lang_token] / total for lang_token in probs}
        
        # ensure every example is used at least once if num_batches is large enough 
        # if num_batches * batch_size >= 210368:
        #     for lang_token in temp:
        #         for example in temp[lang_token]:
        #             self.examples.append(example)
        #             if len(self.examples) % 10000 == 0:
        #                 print(f'Loaded {len(self.examples)}/{num_batches} batches of {split}.')
        
        # sample from pmf until num_batches examples are in self.examples
        lang_token_list = list(c2t.values())
        lang_token_list.remove('spa_Latn')
        weights = [probs[lang_token] for lang_token in lang_token_list]
        for i in range(num_batches):
            
            # randomly select a language based on pmf
            lang_token = choices(lang_token_list, weights=weights)[0]
            
            # randomly select an example of that language
            self.examples.append(choice(temp[lang_token]))
            
            if len(self.examples) % 1000 == 0:
                print(f'Loaded {len(self.examples)} batches of {split}.')
        
        del temp

    def __getitem__(self, idx) -> Union[tuple[list[str], list[str], str], BatchEncoding]:
        
        """
            Returns the batch at the given index: (es_texts, other_texts, lang_token).
            
            Parameters:
            - idx (int): The index of the batch to return.
            
            Returns:
            - tuple[list[str], list[str], str]: The batch at the given index.
            A list of Spanish sentences, a list of sentences in a target language, and the target language token.
        """
        
        if not self.get_tokenized:
            return self.examples[idx]
        
        es_texts, other_texts, lang_token = self.examples[idx]
        
        assert self.tokenizers[lang_token]._src_lang == 'spa_Latn'
        assert self.tokenizers[lang_token].tgt_lang == lang_token
        
        tokenized_batch = self.tokenizers[lang_token](
            text=es_texts,
            text_target=other_texts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_length
        )

        return tokenized_batch

    def __len__(self) -> int:
        
        """
            Returns the number of examples in the dataset.
            
            Returns:
            - int: The number of examples in the dataset.
        """
        
        return len(self.examples)

class SuppDataset(Dataset):
    
    """
        This dataset encapsulates supplementary training data.
        
        Parameters:
        - split: str
            The name of the split ('good_supp' or 'bad_supp').
        - files: list[str]
            A list of file paths to the supplementary data.
        - batch_size: int
            The number of examples in each batch.
        - num_batches: int
            The number of batches to load.
        
    """
    
    def __init__(
        self,
        split: str,
        files: list[str],
        batch_size: int,
        num_batches: int,
        get_tokenized: bool,
        max_length: int
    ):
        
        super().__init__()
        
        # tokenize during getitem or not
        self.get_tokenized = get_tokenized
        if self.get_tokenized:
            self.tokenizers = dict.fromkeys(c2t.values())
            self.max_length = max_length
            for lang_token in self.tokenizers:
                self.tokenizers[lang_token] = make_tokenizer(lang_token, 'spa_Latn', self.max_length)
        
        # list of tokenized batches, each batch is the same target language
        # but different batches may differ in target language
        # since they are pre-tokenized, they are ready to be used in training without worrying
        # about which tokenizer to use
        self.examples = []
        
        # lists to be sampled from later to build self.examples
        temp = dict.fromkeys(c2t.values())
        temp.pop('spa_Latn', None)
        for lang_token in temp:
            temp[lang_token] = []
        
        # process each file (language) and add list of examples to temp
        for file in files:

            # tokenizer from spanish to this language
            lang_token = c2t[path.basename(file).split('.')[0]]
            # tokenizer = tokenizers[lang_token]
            # assert tokenizer._src_lang == 'spa_Latn'
            # assert tokenizer.tgt_lang == lang_token
            
            # lists of spanish and other language sentences
            es_batch = []
            other_batch = []
            
            lines = open(file, 'r', encoding='utf-8').readlines()
            for i in range(len(lines)):
                
                strip_line = lines[i].strip()
                if not strip_line:
                    continue
                split_line = strip_line.split('\t')
                if len(split_line) != 2:
                    continue
                
                # accumulate a batch of es sentences and other language sentences
                es_batch.append(split_line[0].strip())
                other_batch.append(split_line[1].strip())
                
                if len(es_batch) == batch_size or i == len(lines) - 1:
                    
                    assert len(es_batch) == len(other_batch)
                    
                    # tokenize a batch and append to temp dict
                    # tokenized = tokenizers[lang_token](
                    #     text = es_batch,
                    #     text_target = other_batch,
                    #     return_tensors = 'pt',
                    #     padding = 'longest',
                    #     truncation = True,
                    #     max_length = max_length
                    # )
                    # temp[lang_token].append(tokenized)
                    temp[lang_token].append((es_batch, other_batch, lang_token))
                    
                    # clear batch accumulation
                    es_batch = []
                    other_batch = []
                    
                    # no way we could need this much data
                    if i >= num_batches * batch_size:
                        break

            print(f'Loaded {len(temp[lang_token]) * batch_size}/{len(lines)} lines of {lang_token}.')

        # build pmf using exponential reweighting
        probs = dict.fromkeys(c2t.values())
        probs.pop('spa_Latn', None)
        for lang_token in probs:
            probs[lang_token] = 0
            try:
                probs[lang_token] = len(temp[lang_token])**(-0.4)
            except ZeroDivisionError:
                pass
        total = sum(probs.values())
        probs = {lang_token: probs[lang_token] / total for lang_token in probs}
        
        # sample from pmf until num_batches examples are in self.examples
        lang_token_list = list(c2t.values())
        lang_token_list.remove('spa_Latn')
        weights = [probs[lang_token] for lang_token in lang_token_list]
        for i in range(num_batches):
            
            # randomly select a language based on pmf
            lang_token = choices(lang_token_list, weights=weights)[0]
            
            # randomly select a batch of examples of that language
            self.examples.append(choice(temp[lang_token]))
            
            if i % 1000 == 999:
                print(f'Loaded {i+1}/{num_batches} batches of {split}.')
        
        del temp

    def __getitem__(self, idx: int) -> Union[tuple[list[str], list[str], str], BatchEncoding]:
        
        """
            Returns the batch at the given index: (es_texts, other_texts, lang_token).
            
            Parameters:
            - idx (int): The index of the batch to return.
            
            Returns:
            - tuple[list[str], list[str], str]: The batch at the given index.
            A list of Spanish sentences, a list of sentences in a target language, and the target language token.
        """
        
        if not self.get_tokenized:
            return self.examples[idx]
        
        es_texts, other_texts, lang_token = self.examples[idx]
        
        assert self.tokenizers[lang_token]._src_lang == 'spa_Latn'
        assert self.tokenizers[lang_token].tgt_lang == lang_token
        
        tokenized_batch = self.tokenizers[lang_token](
            text=es_texts,
            text_target=other_texts,
            return_tensors='pt',
            padding='longest',
            truncation=True,
            max_length=self.max_length
        )

        return tokenized_batch

    def __len__(self) -> int:
        
        """
            Returns the number of examples in the dataset.
            
            Returns:
            - int: The number of examples in the dataset.
        """
        
        return len(self.examples)

class DevSet(Dataset):
    
    """
        This dataset encapsulates the development set.
        
        Parameters:
        - batch_size (int): The number of examples in each batch.
        - num_batches (int): The number of batches to load.
        - max_length (int): The maximum length of a sequence.
        - lang_code (str): The language code of the target language.
        - use_tgts (bool): Whether to use target sentences.
        - get_tokenized (bool): Whether to tokenize the data.
    """
    
    def __init__(
        self,
        batch_size: int,
        num_batches: int,
        max_length: int,
        lang_code: str,
        use_tgts: bool,
        get_tokenized: bool
    ):
        
        super().__init__()
        
        self.examples = []
        self.lang_code = lang_code
        self.lang_token = c2t[lang_code]
        
        tokenizers = dict.fromkeys(c2t.values())
        
        for lang_token in tokenizers:
            tokenizers[lang_token] = make_tokenizer(lang_token, 'spa_Latn', max_length)
        
        self.tokenizer = tokenizers[self.lang_token]
        assert self.tokenizer._src_lang == 'spa_Latn'
        assert self.tokenizer.tgt_lang == self.lang_token
        
        file_path = path.join('proj_data_final', 'dev', lang_code+'.tsv')
        lines = open(file_path, 'r', encoding='utf-8').readlines()
        
        es_batch = []
        if use_tgts:
            other_batch = []
            
        # count = 0
        
        for i in range(len(lines)):
            
            strip_line = lines[i].strip()
            if not strip_line:
                split_line = ['', '']
            else:
                split_line = strip_line.split('\t')
                if len(split_line) == 1:
                    split_line = [''] + split_line
                elif len(split_line) == 0:
                    split_line = ['', '']
                elif len(split_line) > 2:
                    split_line = split_line[:2]
            
            es_batch.append(split_line[0].strip())
            if use_tgts:
                other_batch.append(split_line[1].strip())
            
            if len(es_batch) == batch_size or i == len(lines) - 1:
                
                if use_tgts:
                    assert len(es_batch) == len(other_batch)
                
                # tokenize a batch and append to temp dict
                if use_tgts:
                    if get_tokenized:
                        assert self.tokenizer._src_lang == 'spa_Latn'
                        assert self.tokenizer.tgt_lang == self.lang_token
                        tokenized = self.tokenizer(
                            text = es_batch,
                            text_target = other_batch,
                            return_tensors = 'pt',
                            padding = 'longest',
                            truncation = True,
                            max_length = max_length
                        )
                        # if self.tokenizer.unk_token_id in tokenized['labels']:
                        #     count += 1
                    else:
                        tokenized = (es_batch, other_batch)
                else:
                    if get_tokenized:
                        assert self.tokenizer._src_lang == 'spa_Latn'
                        assert self.tokenizer.tgt_lang == self.lang_token
                        tokenized = self.tokenizer(
                            text = es_batch,
                            return_tensors = 'pt',
                            padding = 'longest',
                            truncation = True,
                            max_length = max_length
                        )
                        # if self.tokenizer.unk_token_id in tokenized['labels']:
                        #     count += 1
                    else:
                        tokenized = es_batch
                self.examples.append(tokenized)
                
                es_batch = []
                if use_tgts:
                    other_batch = []
            
            if i % 1000 == 999:
                print(f'Loaded {i+1}/{len(lines)} lines for {self.lang_token}.')
            if num_batches is not None and i >= num_batches:
                break
            
        # print(lang_code, str(count), str(len(self.examples)))
                
    def __getitem__(self, idx) -> Union[BatchEncoding, tuple[list[str], list[str]], list[str]]:
        
        """
            Returns the example at the given index.
            
            Parameters:
            - idx (int): The index of the example to return.
            
            Returns
            - Union[transformers.tokenization_utils_base.BatchEncoding, tuple[list[str], list[str]], list[str]]:
                The example at the given index.
        """
        
        return self.examples[idx]

    def __len__(self) -> int:
        
        """
            Returns the number of examples in the dataset.
            
            Returns:
            - int: The number of examples in the dataset.
        """
        
        return len(self.examples)
    
class TestSet(Dataset):
    
    """
        This dataset encapsulates the development set.
        
        Parameters:
        - batch_size (int): The number of examples in each batch.
        - num_batches (int): The number of batches to load.
        - max_length (int): The maximum length of a sequence.
        - lang_code (str): The language code of the target language.
        - use_tgts (bool): Whether to use target sentences.
        - get_tokenized (bool): Whether to tokenize the data.
    """
    
    def __init__(
        self,
        batch_size: int,
        num_batches: int,
        max_length: int,
        lang_code: str,
        use_tgts: bool,
        get_tokenized: bool
    ):
        
        super().__init__()
        
        self.examples = []
        self.lang_code = lang_code
        self.lang_token = c2t[lang_code]
        
        tokenizers = dict.fromkeys(c2t.values())
        
        for lang_token in tokenizers:
            tokenizers[lang_token] = make_tokenizer(lang_token, 'spa_Latn', max_length)
        
        self.tokenizer = tokenizers[self.lang_token]
        assert self.tokenizer._src_lang == 'spa_Latn'
        assert self.tokenizer.tgt_lang == self.lang_token
        
        file_path = path.join('proj_data_final', 'test', lang_code+'.txt')
        lines = open(file_path, 'r', encoding='utf-8').readlines()
        
        es_batch = []
        assert use_tgts == False
            
        # count = 0
        
        for i in range(len(lines)):
         
            es_batch.append(lines[i].strip())
            
            if len(es_batch) == batch_size or i == len(lines) - 1:
                if get_tokenized:
                    assert self.tokenizer._src_lang == 'spa_Latn'
                    assert self.tokenizer.tgt_lang == self.lang_token
                    tokenized = self.tokenizer(
                        text = es_batch,
                        return_tensors = 'pt',
                        padding = 'longest',
                        truncation = True,
                        max_length = max_length
                    )
                    # if self.tokenizer.unk_token_id in tokenized['labels']:
                    #     count += 1
                else:
                    tokenized = es_batch
                self.examples.append(tokenized)
                
                es_batch = []
            
            if i % 1000 == 999:
                print(f'Loaded {i+1}/{len(lines)} lines for {self.lang_token}.')
            if num_batches is not None and i >= num_batches:
                break
            
        # print(lang_code, str(count), str(len(self.examples)))
                
    def __getitem__(self, idx) -> Union[BatchEncoding, tuple[list[str], list[str]], list[str]]:
        
        """
            Returns the example at the given index.
            
            Parameters:
            - idx (int): The index of the example to return.
            
            Returns
            - Union[transformers.tokenization_utils_base.BatchEncoding, tuple[list[str], list[str]], list[str]]:
                The example at the given index.
        """
        
        return self.examples[idx]

    def __len__(self) -> int:
        
        """
            Returns the number of examples in the dataset.
            
            Returns:
            - int: The number of examples in the dataset.
        """
        
        return len(self.examples)

def get_data_loader(
    split: str,
    batch_size: int,
    num_batches: int,
    max_length: int,
    lang_code: Union[str, None],
    shuffle: bool,
    num_workers: int,
    use_tgts: bool,
    get_tokenized: bool
) -> Union[DataLoader, list[DataLoader]]:
    
    """
        Returns a DataLoader (or list of) for the specified split and language code.
        Gives all languages if lang_code is None.
        Returns a list of DataLoaders if split is 'dev' in order to keep languages separate.
        
        Parameters:
        - split (str): The name of the split ('train' or 'good_supp' or 'bad_supp' or 'dev').
        - batch_size (int): The number of examples in each batch.
        - num_batches (int): The number of batches to load.
        - max_length (int): The maximum length of a sequence.
        - lang_code Union[str, None]: The language code of the target language.
        - shuffle (bool): Whether to shuffle the data.
        - num_workers (int): The number of workers to use for loading data.
        - use_tgts (bool): Whether to use target sentences.
        
        Returns:
        - Union[DataLoader, list[DataLoader]]: DataLoader (or list of) for specified split and language code.
    """
    
    if split not in ('dev', 'test'):
        split_loc = path.join('proj_data_final', split)
        files = [path.join(split_loc, f) for f in listdir(split_loc) if f.endswith('.tsv')]
        if lang_code is not None:
            files = [f for f in files if lang_code in f]
        if split != 'train':
            return DataLoader(
                SuppDataset(split, files, batch_size, num_batches, get_tokenized, max_length),
                batch_size=1, # batches are already created, so just load one at a time
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn # remove the extra dimension that DataLoader adds
            )
        else:
            return DataLoader(
                TrainDataset(split, files, batch_size, num_batches, get_tokenized, max_length),
                batch_size=1, # batches are already created, so just load one at a time
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=collate_fn # remove the extra dimension that DataLoader adds
            )
    elif split == 'dev':
        if lang_code is not None:
            return [DataLoader(
                DevSet(batch_size, num_batches, max_length, lang_code, use_tgts, get_tokenized),
                batch_size=1, # batches are already created, so just load one at a time
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn # remove the extra dimension that DataLoader adds
            )]
        else:
            return [DataLoader(
                DevSet(batch_size, num_batches, max_length, l, use_tgts, get_tokenized),
                batch_size=1, # batches are already created, so just load one at a time
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn # remove the extra dimension that DataLoader adds
            ) for l in c2t.keys() if l != 'es']
    elif split == 'test':
        if lang_code is not None:
            return [DataLoader(
                TestSet(batch_size, num_batches, max_length, lang_code, use_tgts, get_tokenized),
                batch_size=1, # batches are already created, so just load one at a time
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn # remove the extra dimension that DataLoader adds
            )]
        else:
            return [DataLoader(
                TestSet(batch_size, num_batches, max_length, l, use_tgts, get_tokenized),
                batch_size=1, # batches are already created, so just load one at a time
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn # remove the extra dimension that DataLoader adds
            ) for l in c2t.keys() if l != 'es']
    else:
        raise ValueError('Invalid split name.')