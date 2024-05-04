import os
from datetime import timedelta
from gc import collect
from time import time

import sentencepiece as spm
import torch
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from matplotlib.pyplot import plot, figure, savefig, grid, legend, title
from torch import optim, save, no_grad
from torch.cuda import empty_cache, memory_allocated
from torch.utils.data import DataLoader

from dataset import TranslationDataset

sp = spm.SentencePieceProcessor()
sp.load(os.path.join('sp', 'unigram_16k.model'))

MAX_SEQ_LEN = 512

def plot_losses(
        losses: list,
        plot_title: str,
        output_str: str
):
    figure()
    plot(
        range(len(losses)),
        losses,
        label='train'
    )
    title(plot_title)
    grid()
    legend()

    if not os.path.exists('plots'):
        os.mkdir('plots')
    savefig(os.path.join('plots', f'{plot_title}_{output_str}.png'))


def collate_fn(batch):
    pad_idx = sp.piece_to_id('<pad>')
    longest_seq_len = 0

    for item in batch:
        if len(item[0]) > longest_seq_len:
            longest_seq_len = len(item[0])
        if len(item[1]) > longest_seq_len:
            longest_seq_len = len(item[1])

    if longest_seq_len > MAX_SEQ_LEN:
        longest_seq_len = MAX_SEQ_LEN

    src_batch = [torch.nn.ConstantPad1d((0, longest_seq_len - len(item[0])), pad_idx)(item[0])[0:MAX_SEQ_LEN] for item
                 in batch]
    tgt_batch = [torch.nn.ConstantPad1d((0, longest_seq_len - len(item[1])), pad_idx)(item[1])[0:MAX_SEQ_LEN] for item
                 in batch]

    free()

    return torch.stack(src_batch), torch.stack(tgt_batch)


def get_data_loader(dataset, batch_size=128, num_workers=0, shuffle=True, pin_memory=True):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory,
                        collate_fn=collate_fn)
    return loader


def free():
    """
        Free memory and print memory usage.
    """

    collect()
    empty_cache()
    # print(f'Memory: {Process().memory_info().rss / (1024 * 1024)} MB')


def eval(model: MambaLMHeadModel, device: str):
    model.eval()

    langs = ['aym', 'bzd', 'cni', 'ctp', 'gn', 'hch', 'nah', 'oto', 'quy', 'shp', 'tar']
    files = []
    for lang in langs:
        files.append(os.path.join('data', 'dev', lang + '.tsv'))

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=sp.piece_to_id('<pad>'))

    with no_grad():
        for file in files:
            ds = TranslationDataset([file], sp)
            loader = get_data_loader(dataset=ds, batch_size=64, num_workers=1, shuffle=False)
            i = 0
            for i, batch in enumerate(loader):
                src, tgt = batch
                outputs = model(src.to(device))[0]

                free()
                loss = loss_fn(
                    outputs.view(-1, outputs.shape[-1]).to(device),
                    tgt.view(-1).to(device)
                )
                item = loss.item()  # log and store loss
                print(f'Batch {i + 1}/{len(loader)} complete, {file} loss: {item}')
                del batch
                del loss
                del item


def train(
        model: MambaLMHeadModel,
        batch_size: int,
        num_workers: int,
        epochs: int,
        optimizer: optim.Optimizer,
        device: str,
        log_freq: int,
        files: list[str],
        output_str: str,
        eval_epochs: bool
):
    dataset = TranslationDataset(files, sp)
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    losses = []
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=sp.piece_to_id('<pad>'))

    for epoch in range(epochs):  # epoch loop
        epoch_start = time()
        print(f'Epoch {epoch + 1} starting...')
        free()
        model.train()  # ensure training mode
        i = 0
        for i, batch in enumerate(loader):  # batch loop
            free()
            src, tgt = batch

            optimizer.zero_grad()  # run training step

            outputs = model(src.to(device))[0]

            free()
            loss = loss_fn(
                outputs.view(-1, outputs.shape[-1]).to(device),
                tgt.view(-1).to(device)
            )
            loss.backward()
            optimizer.step()
            item = loss.item()  # log and store loss

            if i % log_freq == log_freq - 1:
                print(f'Batch {i + 1}/{len(loader)} complete, loss: {item}')

            losses.append(item)

            # free memory
            del item
            del batch
            del outputs
            del loss
            collect()
            empty_cache()

        # free memory and log
        optimizer.zero_grad()
        free()
        print(f'Epoch {epoch + 1} train complete in {str(timedelta(seconds=time() - epoch_start))}.\n')

        # save checkpoint
        print('Saving checkpoint...')
        free()
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        ckpts_dir = 'ckpts256_3_16'
        if not os.path.exists(ckpts_dir):
            os.mkdir(ckpts_dir)
        save(
            checkpoint,
            os.path.join(
                ckpts_dir,
                f'checkpoint{epoch + 1}_{output_str}.pth'
            )
        )
        del checkpoint
        free()

        if eval_epochs:
            eval(model, device)
            free()

        print('Done.\n')
    del loader  # free memory
    free()

    return losses


def main():
    """ HYPERPARAMETERS """
    log_freq = 2048  # frequency of logging in batches
    eval_epochs = True  # run metrics on dev set
    num_workers = 1  # number of workers for data loader

    batch_size = 128

    lr = 1e-3
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print('\nLoading model...')
    config = MambaConfig(d_model=256, n_layer=3, vocab_size=sp.vocab_size())
    model = MambaLMHeadModel(config, device='cuda').to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {param_count:_}')
    print(f'Model size on GPU: {memory_allocated(device=device) / 1024 ** 3:.2f} GB.\n')

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )

    train_losses = []  # set up for plotting
    dev_losses = []

    langs = ['aym', 'bzd', 'cni', 'ctp', 'gn', 'hch', 'nah', 'oto', 'quy', 'shp', 'tar']
    files = []
    for lang in langs:
        files.append(os.path.join('data', 'stage3', lang + '.tsv'))
        files.append(os.path.join('data', 'stage2', lang + '.tsv'))
        files.append(os.path.join('data', 'stage1', lang + '.tsv'))

    epochs = 5
    output_str = 'mamba_256_3_16000_stages_1_2_3'

    train(model, batch_size, num_workers, epochs, optimizer, device, log_freq, files, output_str,
                         eval_epochs)

    files = []
    for lang in langs:
        files.append(os.path.join('data', 'train', lang + '.tsv'))
        files.append(os.path.join('data', 'extra', lang + '.tsv'))

    epochs = 25
    output_str = 'mamba_256_3_16000_stages_2_3'

    train(model, batch_size, num_workers, epochs, optimizer, device, log_freq, files, output_str,
          eval_epochs)


if __name__ == '__main__':
    main()
