# nllb-finetune

This code can be used to fine-tune NLLB-200 on the [AmericasNLP 2024 Shared Task 1 data](https://github.com/AmericasNLP/americasnlp2024/tree/master/ST1_MachineTranslation), a shared task organized by the [AmericasNLP Workshop](https://turing.iimas.unam.mx/americasnlp/index.html).

## Formatting data correctly

The scripts expect all data to be in .tsv format, with all lines in the following format:

`<Spanish text>\t<target language text>\n`

All data in any given file should have the same target language, and the file should be named `<language_code>.tsv`.

One file should exist per language in each of the four subdirectories of the `data` directory:

- `stage1` (supplemental MT/synthetic data - 11 files: aym.tsv, bzd.tsv, ...)
- `stage2` (supplemental non-MT/synthetic data - 11 files: aym.tsv, bzd.tsv, ...)
- `stage3` (official training data - 11 files: aym.tsv, bzd.tsv, ...)
- `dev` (official dev data - 11 files: aym.tsv, bzd.tsv, ...)

`assert_form.py` can be run to check that the `data` directory is correctly formatted.

## Setting up to train

`discover_seq_length.py` is a useful script to identify possible choices for choosing a maximum sequence length for training. It will print the lengths of all sequences in the development set (greatest to least), according to the NLLB tokenizer, and the longest length for easy reference.

`requirements.txt` contains the versions of `transformers` and `torch` that were used to train the model.

`get_data_loader.py` and `no_sample_data_loader.py` can be used to retrieve data. The former will use over- and under-sampling by language when retrieving the training and supplementary datasets, an idea we abandoned when deciding how best to fine-tune NLLB-200. The latter was used to retrieve the training data for the final model. They function identically for the dev and test sets.

## Training

`train.py` is the script to train the model. Hyperparameters can be modified in the `main()` function at the bottom of the script. The script will load tokenizers (`make_tokenizer.py`) with the correct target language set, load a model or checkpoint, and train it as specified according to the hyperparameters. It will save model and optimizer checkpoints to `outputs/ckpts`, loss plots to `outputs/plots`, and decode the dev set for each checkpoint to `outputs/translations`.

## Decoding the dev and test sets

Decoding the dev set can also be done using `decode_dev.py`. This script can load selected checkpoints if problems occurr with the decoding process in `train.py`. Checkpoints to load can be specified in a file named `bad_ckpts.txt`.

Decoding the test set can be done using `decode_test.py`. Test data is expected to be located in `data/test`, with each file named as `<language_code.txt>`. Decoded outputs are saved to `outputs/translations_test`.

## Evaluating outputs

The repository expects the [shared task evaluation script](https://github.com/AmericasNLP/americasnlp2024/blob/master/ST1_MachineTranslation/evaluate.py), `evaluate.py`, to be in the root directory.

`evaluate_models.py` will run metrics using `evaluate.py` on every checkpoint's dev set outputs, as are found in `outputs/translations`. It will save the results to `outputs/reports`.

Reports can then be analyzed on a per-language basis as needed using `reports.py`, which will print some useful information garnered from the reports.

