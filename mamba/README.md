# Mamba

This code can be used to train a Mamba-based neural network on the [AmericasNLP 2024 Shared Task 1 data](https://github.com/AmericasNLP/americasnlp2024/tree/master/ST1_MachineTranslation), a shared task organized by the [AmericasNLP Workshop](https://turing.iimas.unam.mx/americasnlp/index.html).

## Formatting data correctly

The scripts expect all data to be in .tsv format, with all lines in the following format:

`<Spanish text>\t<target language text>\n`

All data in any given file should have the same target language, and the file should be named `<language_code>.tsv`.

One file should exist per language in each of the four subdirectories of the `data` directory:

- `stage1` (supplemental MT/synthetic data - 11 files: aym.tsv, bzd.tsv, ...)
- `stage2` (supplemental non-MT/synthetic data - 11 files: aym.tsv, bzd.tsv, ...)
- `stage3` (official training data - 11 files: aym.tsv, bzd.tsv, ...)
- `dev` (official dev data - 11 files: aym.tsv, bzd.tsv, ...)

## Training

`train.py` is the script to train the model. Hyperparameters can be modified in the `main()` function.
The script loads a SentencePiece tokenizer trained with the `sp.y` and trains a from-scratch model through 5 epochs on all data (combined `stage 1`, `stage 2`, and `stage 3`) followed by 25 epochs on combined `stage 2` and `stage 3` data.
Model checkpoints are saved to a `ckpts` directory.

## Decoding the dev and test sets

Decoding the dev set and test set can be done using `decode_dev.py` and  `decode_test.py`, respectively.

## Evaluating outputs

Evaluation on the decoded dev output can be done using the [shared task evaluation script](https://github.com/AmericasNLP/americasnlp2024/blob/master/ST1_MachineTranslation/evaluate.py), `evaluate.py`, provided by task organizers.
