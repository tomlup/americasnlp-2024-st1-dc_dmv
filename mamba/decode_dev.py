import os
from gc import collect
import torch

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache
from torch import no_grad, load, manual_seed

from train import get_data_loader
from dataset import TranslationDataset
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load(os.path.join('sp', 'unigram_16k.model'))

MAX_SEQ_LEN = 512


def free():
    collect()
    empty_cache()


def main():
    freeze_support()

    # device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42)

    print('\nLoading model...')
    free()
    config = MambaConfig(d_model=256, n_layer=3, vocab_size=sp.vocab_size())
    model = MambaLMHeadModel(config, device='cuda').to(device)
    free()
    print('Model loaded.\n')

    print('Loading dev data...')
    free()
    langs = ['aym', 'bzd', 'cni', 'ctp', 'gn', 'hch', 'nah', 'oto', 'quy', 'shp', 'tar']
    files = []
    for lang in langs:
        files.append(os.path.join('data', 'dev', lang + '.tsv'))
    free()
    print('Dev data loaded.\n')

    tr_dir = os.path.join('dev-translations')
    if not os.path.exists(tr_dir):
        os.mkdir(tr_dir)

    ckpts = [
        'checkpoint25_mamba_256_3_16000_ALL.pth'
    ]

    for ckpt in ckpts:

        model_tr_dir = os.path.join(tr_dir)

        print(f'Loading checkpoint {ckpt}...')
        free()
        file_path = os.path.join('ckpts', ckpt)
        try:
            checkpoint = load(file_path)
        except Exception as e:
            print(f'Failed to load checkpoint {ckpt}.')
            print(e)
            continue
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        free()
        print(f'Checkpoint {ckpt} loaded.\n')

        with no_grad():
            model.eval()

            for lang in langs:
                translations = []

                file = os.path.join('data', 'dev', lang + '.tsv')
                ds = TranslationDataset([file], sp)
                loader = get_data_loader(dataset=ds, batch_size=1, num_workers=1, shuffle=False)

                for i, batch in enumerate(loader):
                    source = batch[0]
                    outputs = model(source.to(device))[0]

                    _, topi = outputs.topk(1)
                    output_ids = topi.squeeze()
                    output_sent = sp.DecodeIds(output_ids.tolist())

                    translations.append(output_sent)

                    if i % 100 == 0:
                        print(f'{i} batches decoded for {lang}.')

                    del outputs
                    free()

                    if not os.path.exists(model_tr_dir):
                        os.mkdir(model_tr_dir)
                    loc = os.path.join(model_tr_dir, lang + '.txt')

                    with open(loc, 'w+', encoding='utf-8') as f:
                        for t in translations:
                            f.write(t + '\n')
                    free()


if __name__ == '__main__':
    main()
