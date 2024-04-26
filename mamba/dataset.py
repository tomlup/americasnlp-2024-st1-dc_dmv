from os import path

import sentencepiece as spm
import torch
from torch.utils.data import Dataset


class SourceOnlyDataset(Dataset):
    """
        This dataset encapsulates the test data.

        Parameters:
        - files (list[str]): A list of file paths to the training data (.tsv).
        - tokenizer (spm.SentencePieceProcessor): A SentencePiece tokenizer.
    """

    def __init__(
            self,
            files: list[str],
            sp: spm.SentencePieceProcessor,
            lang: str
    ):

        super().__init__()
        self.sp = sp
        self.sources = []
        self.lang_token = '<' + lang + '>'

        self.SOS_IDX = self.sp.piece_to_id('<s>')
        self.EOS_IDX = self.sp.piece_to_id('</s>')

        # process each file (language)
        for file in files:
            lines = open(file, 'r', encoding='utf-8').readlines()
            for i in range(len(lines)):
                strip_line = lines[i].strip()
                if not strip_line:
                    continue

                self.sources.append(strip_line)

                if i % 1000 == 999:
                    print(f'Loaded {i + 1}/{len(lines)} lines of {self.lang_token}.')

    def __getitem__(self, idx):  # -> Tensor

        src_sent = self.sources[idx]

        src_ids = [self.SOS_IDX, self.sp.piece_to_id(self.lang_token)] + self.sp.EncodeAsIds(src_sent) + [self.EOS_IDX]

        return torch.tensor(src_ids, dtype=torch.int64)

    def __len__(self) -> int:

        return len(self.sources)


class TranslationDataset(Dataset):
    """
        This dataset encapsulates the parallel translation data.

        Parameters:
        - files (list[str]): A list of file paths to the training data (.tsv).
        - tokenizer (spm.SentencePieceProcessor): A SentencePiece tokenizer.
    """

    def __init__(
            self,
            files: list[str],
            sp: spm.SentencePieceProcessor
    ):

        super().__init__()
        self.sp = sp
        self.sources = []
        self.targets = []
        self.langs = []

        self.SOS_IDX = self.sp.piece_to_id('<s>')
        self.EOS_IDX = self.sp.piece_to_id('</s>')

        # process each file (language)
        for file in files:
            lang_token = path.basename(file).split('.')[0]
            lang_token = '<' + lang_token + '>'

            lines = open(file, 'r', encoding='utf-8').readlines()
            for i in range(len(lines)):
                strip_line = lines[i].strip()
                if not strip_line:
                    continue
                split_line = strip_line.split('\t')
                if len(split_line) != 2:
                    continue

                self.sources.append(split_line[0].strip())
                self.targets.append(split_line[1].strip())
                self.langs.append(lang_token)

                if i == len(lines) - 1:
                    assert len(self.sources) == len(self.targets)

                if i % 1000 == 999:
                    print(f'Loaded {i + 1}/{len(lines)} lines of {lang_token}.')

    def __getitem__(self, idx):  # -> Tensor

        src_sent = self.sources[idx]
        tgt_sent = self.targets[idx]
        lang_token = self.langs[idx]

        src_ids = [self.SOS_IDX, self.sp.piece_to_id(lang_token)] + self.sp.EncodeAsIds(src_sent) + [self.EOS_IDX]
        tgt_ids = [self.SOS_IDX] + self.sp.EncodeAsIds(tgt_sent) + [self.EOS_IDX]

        return torch.tensor(src_ids, dtype=torch.int64), torch.tensor(tgt_ids, dtype=torch.int64)

    def __len__(self) -> int:

        return len(self.sources)
