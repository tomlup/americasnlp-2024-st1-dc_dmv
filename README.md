# DC_DMV's Submission to AmericasNLP Shared Task 1: Machine Translation Systems for Indigenous Languages

This repository contains code used to train our models as part of the [AmericasNLP 2024 Shared Task 1](https://github.com/AmericasNLP/americasnlp2024/tree/master/ST1_MachineTranslation), a shared task organized by the [AmericasNLP Workshop](https://turing.iimas.unam.mx/americasnlp/index.html).
Our submission to this shared task included two approaches: fine-tuning [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) and training from scratch a [Mamba](https://github.com/state-spaces/mamba/tree/main)-based neural network. Code for each approach can be found in the folders `mamba` and `nllb` contained in this repository.

Our paper can be found here: [Experiments in Mamba Sequence Modeling and NLLB-200 Fine-Tuning for Low Resource Multilingual Machine Translation](https://aclanthology.org/2024.americasnlp-1.22/)

## Acknowledgements

We would like to thank Dr. Kenton Murray of Johns Hopkins University for his guidance and support throughout the project. We would also like to thank the AmericasNLP 2024 Shared Task 1 organizers for providing the data and evaluation script.
