import os
from gc import collect
import torch

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache
from torch import no_grad, load, manual_seed
from torch.utils.data import DataLoader

from train import get_data_loader
from dataset import SourceOnlyDataset
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load(os.path.join('sp', 'unigram_16k.model'))

MAX_SEQ_LEN = 512


def free():
    collect()
    empty_cache()

def collate_fn(batch):
    pad_idx = sp.piece_to_id('<pad>')
    longest_seq_len = 0

    for item in batch:
        if len(item) > longest_seq_len:
            longest_seq_len = len(item)

    if longest_seq_len > MAX_SEQ_LEN:
        longest_seq_len = MAX_SEQ_LEN

    src_batch = [torch.nn.ConstantPad1d((0, longest_seq_len - len(item)), pad_idx)(item)[0:MAX_SEQ_LEN] for item
                 in batch]
    free()

    return torch.stack(src_batch)

def get_data_loader(dataset, batch_size=128, num_workers=0, shuffle=True, pin_memory=True):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
                        collate_fn=collate_fn)
    return loader

def main():
    freeze_support()

    # device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42)

    overfit = False
    num_workers = 1
    batch_size = 1

    print('\nLoading model...')
    free()
    config = MambaConfig(d_model=256, n_layer=3, vocab_size=sp.vocab_size())
    model = MambaLMHeadModel(config, device='cuda').to(device)
    # print('Loading from checkpoint...')
    #
    # model.load_state_dict(checkpoint['model_state_dict'])
    # print(f'Model {ckpt_string} loaded.')
    free()
    print('Model loaded.\n')

    print('Loading test data...')
    free()
    langs = ['ctp', 'tar']
    free()
    print('Dev data loaded.\n')

    tr_dir = os.path.join('test_translations')
    if not os.path.exists(tr_dir):
        os.mkdir(tr_dir)

    # ckpts = os.listdir(os.path.join('ckpts256_3_16'))
    # ckpts = os.listdir(os.path.join('ckpts512_3_16'))
    ckpts = [
        #'checkpoint48_mamba_256_3_16000_EXTRAANDTRAIN.pth'
        'checkpoint25_mamba_256_3_16000_ALL.pth',
        # 'checkpoint24_mamba_256_3_16000_EXTRAANDTRAIN.pth'
    ]

    for ckpt in ckpts:

        model_tr_dir = os.path.join(tr_dir, ckpt[:-4])

        print(f'Loading checkpoint {ckpt}...')
        free()
        file_path = os.path.join('ckpts256_3_16', ckpt)
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
                if lang == 'ctp':
                    files = [os.path.join('data', 'test', 'ctp_test.es')]
                else:
                    files = [os.path.join('data', 'test', 'test.es')]

                translations = []
                ds = SourceOnlyDataset(files, sp, lang=lang)
                loader = get_data_loader(dataset=ds, batch_size=1, num_workers=1, shuffle=False)

                for i, batch in enumerate(loader):
                    source = batch
                    # print(source)
                    outputs = model(source.to(device))[0]
                    print(outputs)
                    _, topi = outputs.topk(1)
                    output_ids = topi.squeeze()
                    output_sent = sp.DecodeIds(output_ids.tolist())

                    translations.append(output_sent)

                    # try:
                    #     assert len(translations) == (i + 1) * batch_size
                    # except:
                    #     print(f'Batch {i} failed for {lang}.')
                    #     print(f'Batch size: {batch_size}, translations length: {len(translations)}.')
                    #     if i == len(loader) - 1:
                    #         print(f'Last batch for {lang}.')
                    # print(f'num translations: {len(translations)}')

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
