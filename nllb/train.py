from os import path, mkdir
from gc import collect
from time import time
from datetime import timedelta
from psutil import Process

from torch.multiprocessing import freeze_support
from torch.cuda import is_available, empty_cache, memory_allocated
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch import manual_seed, no_grad, save, load

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from matplotlib.pyplot import plot, figure, savefig, grid, legend, title

#from get_data_loader import get_data_loader
from no_sample_data_loader import get_data_loader

from make_tokenizer import make_tokenizer, c2t, t2i

# potential issues:
# hf surgery - model seems to have right size when loading up checkpoints
# data loader - maybe sampling is wrong, some things occur multiple times in small set
# bad supp might just be garbage and should be ignored
# maybe no random sampling is good, or at least ensuring all good supp and train are included

def free():
    
    """
        Free memory and print memory usage.
    """
    
    collect()
    empty_cache()
    print(f'Memory: {Process().memory_info().rss / (1024 * 1024)} MB')

def plot_losses(
    bad: list,
    good: list,
    train: list,
    plot_title: str,
    output_str: str
):
    
    """
        Plot losses for each split.
        
        Parameters:
        - bad (list): Losses for bad_supp.
        - good (list): Losses for good_supp.
        - train (list): Losses for train.
        - plot_title (str): Title of the plot.
    """
    
    figure()
    plot(
        range(len(bad)),
        bad,
        label='bad_supp'
    )
    plot(
        range(len(bad), len(bad) + len(good)),
        good,
        label='good_supp'
    )
    plot(
        range(len(bad) + len(good), len(bad) + len(good) + len(train)),
        train,
        label='train'
    )
    title(plot_title)
    grid()
    legend()
    plots_dir = path.join('outputs', 'plots')
    if not path.exists(plots_dir):
        mkdir(plots_dir)
    savefig(path.join(plots_dir, f'{plot_title}_{output_str}.png'))

def decode(
    ckpt_str: str,
    model: AutoModelForSeq2SeqLM,
    dev_loaders: list,
    device: str,
    max_length: int
):
    tr_dir = path.join('outputs', 'translations')
    if not path.exists(tr_dir):
        mkdir(tr_dir)
        
    model_tr_dir = path.join(tr_dir, ckpt_str[:-4])
    if not path.exists(model_tr_dir):
        mkdir(model_tr_dir)
    
    with no_grad():
        model.eval()
        for dev_loader in dev_loaders:
            
            lang_code = dev_loader.dataset.lang_code
            lang_token = dev_loader.dataset.lang_token
            tokenizer = dev_loader.dataset.tokenizer
            
            translations = []
            
            for i, batch in enumerate(dev_loader):
                
                outputs = model.generate(
                    **batch.to(device),
                    forced_bos_token_id=t2i[lang_token],
                    max_length=max_length,
                    num_beams=4,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                translations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
                
                if i % 100 == 0:
                    print(f'{i} batches decoded for {lang_code}.')
                
            loc = path.join(model_tr_dir, lang_code + '.txt')
                
            with open(loc, 'w+', encoding='utf-8') as f:
                for t in translations:
                    f.write(t + '\n')

def train(
    loader_name: str,
    tokenizers: dict[str, AutoTokenizer],
    batch_size: int,
    num_batches: int,
    max_length: int,
    lang_code: str,
    num_workers: int,
    epochs: int,
    model: AutoModelForSeq2SeqLM,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: str,
    dev_num_batches: int,
    ckpt: bool,
    output_str: str,
    do_dev: bool,
    log_freq: int,
    get_tokenized: bool,
    overfit: bool
) -> tuple[list, list]:
    
    """
        Train the model on the specified data loader.
        
        Parameters:
        - loader_name (str): Name of the data loader. One of 'bad_supp', 'good_supp', 'train'.
        - tokenizers (dict[str, AutoTokenizer]): Tokenizers for each language.
        - batch_size: (int): Batch size.
        - num_batches (int): Number of batches to train on.
        - max_length: (int): Maximum length of the input sequences.
        - lang_code (str): Language code. Set to None for all languages.
        - num_workers (int): Number of workers for the data loader.
        - epochs: (int): Number of epochs.
        - model (AutoModelForSeq2SeqLM): Model to train.
        - optimizer (optim.Optimizer): Optimizer.
        - device (str): Device.
        - dev_num_batches (int): Number of batches to evaluate on.
        - ckpt (bool): Whether to save checkpoints.
        - output_str (str): Output string.
        - do_dev (bool): Whether to evaluate on dev. Ignored except for 'train' split.
        - log_freq (int): Frequency of logging in batches.
        - get_tokenized (bool): Whether to return tokenized data.
            
        Returns:
        - tuple[list, list]: Train losses and dev losses.
    """
    
    print('Loading data...') # retrieve appropriate data loader
    free()
    print(loader_name, batch_size, num_batches, max_length, lang_code, num_workers, get_tokenized)
    train_loader = get_data_loader(
        split=loader_name,
        batch_size=batch_size,
        num_batches=num_batches,
        max_length=max_length,
        lang_code=lang_code,
        shuffle=True,
        num_workers=num_workers,
        use_tgts=True, # ignored
        get_tokenized=get_tokenized
    )
    free()
    print('Data loaded.\n')
    
    # print('loader:', train_loader, len(train_loader), '\n')
    # for j, batch in enumerate(train_loader):
    #     if j % 5000 == 0:
    #         print(j, batch[0][0][:10], batch[1][0][:10], batch[2])
    #         print()
    # exit()
    
    train_losses = [] # set up for plotting
    dev_losses = []
    
    for epoch in range(epochs): # epoch loop
        
        ckpt_str = f'checkpoint{epoch+1}_{loader_name}_{output_str}.pth'
        
        epoch_start = time()
        print(f'Epoch {epoch+1} starting...')
        free()
        model.train() # ensure training mode
        for i, batch in enumerate(train_loader): # batch loop
            if not get_tokenized:
                es_texts, other_texts, lang_token = batch # unpack batch and lang token and tokenize
                assert tokenizers[lang_token].tgt_lang == lang_token
                assert tokenizers[lang_token]._src_lang == 'spa_Latn'
                tokenized_batch = tokenizers[lang_token](
                    text=es_texts,
                    text_target=other_texts,
                    return_tensors='pt',
                    padding='longest',
                    truncation=True,
                    max_length=max_length
                )
            else:
                tokenized_batch = batch
            optimizer.zero_grad() # run training step
            outputs = model(**tokenized_batch.to(device))
            loss = outputs.loss
            loss.backward()
            #clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            #scheduler.step()
            item = loss.item() # log and store loss
            if i % log_freq == log_freq - 1:
                print(f'Batch {i+1}/{len(train_loader)} complete, loss: {item}')
            train_losses.append(item)
            del item # free memory
            del batch
            del outputs
            del loss
            collect()
            empty_cache()
        optimizer.zero_grad() # free memory and log
        free()
        print(f'Epoch {epoch+1} train complete in {str(timedelta(seconds=time()-epoch_start))}.\n')
        
        if do_dev: # evaluate on dev
            print('Loading dev data...') # retrieve dev data loader
            free()
            dev_loaders = get_data_loader(
                split='dev',
                batch_size=batch_size,
                num_batches=dev_num_batches,
                max_length=max_length,
                lang_code=lang_code,
                shuffle=False, # ignored
                num_workers=num_workers,
                use_tgts=True, # for dev loss
                get_tokenized=get_tokenized # ignored
            )
            free()
            print('Dev data loaded.\n')

            print(f'Epoch {epoch+1} eval starting...')
            free()
            model.eval() # ensure evaluation mode
            with no_grad(): # no gradients for evaluation
                # evaluate on each dev data loader - one for each language
                for dev_loader in dev_loaders: # dev loop
                    lang_token = dev_loader.dataset.lang_token # fetch lang token
                    for i, batch in enumerate(dev_loader): # dev batch loop
                        outputs = model(**batch.to(device)) # pretokenized batches
                        loss = outputs.loss
                        item = loss.item() # log and store loss
                        if i % log_freq == log_freq - 1:
                            msg = f'Dev batch {i+1}/{len(dev_loader)} complete'
                            msg += f' (lang={lang_token}), loss: {item}'
                            print(msg)
                        dev_losses.append(item)
                        del item # free memory
                        del batch
                        del outputs
                        del loss
                        collect()
                        empty_cache() 
            free()
            print(f'Finished computing losses. Running inference.\n')
            decode(
                ckpt_str,
                model,
                dev_loaders,
                device,
                max_length
            )
            if not overfit:
                del dev_loaders # free memory and log
            free()
            print(f'Epoch {epoch+1} eval complete.\n')

        if ckpt: # save checkpoint
            print('Saving checkpoint...')
            free()
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            ckpts_dir = path.join('outputs', 'ckpts')
            if not path.exists(ckpts_dir):
                mkdir(ckpts_dir)
            save(
                checkpoint,
                path.join(
                    ckpts_dir,
                    ckpt_str
                )
            )
            del checkpoint
            free()
            print('Done.\n')
            
    del train_loader # free memory
    free()
        
    return train_losses, dev_losses # return losses for plotting

def main():
    
    """ HYPERPARAMETERS """ # TODO: search for optimal hyperparameters
    load_ckpt         = True                              # whether to load a checkpoint
    ckpt_file_name    = 'checkpoint10_train_False_4_0_2500_3_None_10_None_1e-05_0.0001.pth'
    overfit           = False                              # overfit on small data to test functionality
    log_freq          = 100     if not overfit else 1      # frequency of logging in batches
    num_workers       = 2                                  # number of workers for data loader
    get_tokenized     = True                               # whether to get tokenized data
    freeze            = False # TODO: decide                # freeze most of the model
    
    batch_size        = 4       if not overfit else 1      # batch size
    max_length        = 384     if not overfit else 16     # maximum length of input sequences
    lang_code         = None    if not overfit else 'aym'  # None for all languages
    
    lr                = 1e-5                               # learning rate
    weight_decay      = 1e-4                               # weight decay
    warmup            = 0.1                                # warmup proportion
    
    bad_epochs        = 0       if not overfit else 0      # num epochs through bad_supp
    do_bad            = False    if not overfit else False # whether to train on bad_supp
    
    good_epochs       = 0       if not overfit else 1      # num epochs through good_supp
    do_good           = False    if not overfit else False  # whether to train on good_supp
    
    train_epochs      = 10      if not overfit else 50     # every training example is guaranteed included:
    
    dev_num_batches   = None    if not overfit else 20     # None for full dev set
    do_dev            = True    if not overfit else True   # whether to evaluate on dev (ignored for supp data)
    ckpt              = True    if not overfit else False  # whether to save checkpoints
    
    bad_num_batches   = int(10_000 / batch_size) if not overfit else 1  # random sampling is used
    good_num_batches  = None if not overfit else 1
    train_num_batches = None if not overfit else 20
    # random sampling for train_num_batches IF train_num_batches * batch_size > 210368
    
    start = time()
    freeze_support() # parallelism for Windows
    #device = 'cpu'
    device = 'cuda' if is_available() else 'cpu'
    manual_seed(42) # set random seed for reproducibility

    print(f'Using {num_workers} workers.')
    
    print('\nLoading model...')
    tokenizers = dict.fromkeys(c2t.values())
    for lang_token in tokenizers: # load tokenizers for each language
        tokenizers[lang_token] = make_tokenizer(lang_token, 'spa_Latn', max_length)
        assert tokenizers[lang_token]._src_lang == 'spa_Latn'
        assert tokenizers[lang_token].tgt_lang == lang_token
        assert len(tokenizers[lang_token]) == 256212
    model_name = 'facebook/nllb-200-distilled-600M'
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        ignore_mismatched_sizes=True
    )
    model.resize_token_embeddings(len(tokenizers['ayr_Latn'])) # resize embeddings
    ckpt_path = path.join('outputs', 'ckpts', ckpt_file_name)
    if load_ckpt:
        print('Loading checkpoint...')
        model.load_state_dict(load(ckpt_path)['model_state_dict'])
    # freeze embeddings and decoder
    if freeze:
        print('Freezing almost everything...')
        for name, param in model.named_parameters():
            param.requires_grad = False
            if 'decoder.layers.11' in name:
                param.requires_grad = True
    print('Model loaded.')
    model.to(device)
    print(f'Using device: {device}.')
    print(f'Model size on {device}: {memory_allocated(device=device) / 1024**3:.2f} GB.\n')
    
    # output string for saving plots and checkpoints
    output_str = f'{freeze}_{batch_size}_{bad_epochs}_{bad_num_batches}_{good_epochs}'
    output_str += f'_{good_num_batches}_{train_epochs}_{train_num_batches}_{lr}_{weight_decay}'
    if load_ckpt:
        output_str += f'_ckpt_{ckpt_file_name[:-4]}'
    if not path.exists('outputs'):
        mkdir('outputs')
    
    optimizer = AdamW( # AdamW optimizer
        model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=weight_decay
    )
    if load_ckpt:
        optimizer.load_state_dict(load(ckpt_path)['optimizer_state_dict'])
    # total_batches = bad_num_batches + good_num_batches + train_num_batches
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=int(total_batches * warmup),
    #     num_training_steps=total_batches
    # )
    scheduler = None
    
    """ TRAINING - BAD SUPP """
    if do_bad:
        print('Training on bad supp...')
        bad_train_losses, bad_dev_losses = train(
            loader_name='bad_supp',
            tokenizers=tokenizers,
            batch_size=batch_size,
            num_batches=bad_num_batches,
            max_length=max_length,
            lang_code=lang_code,
            num_workers=num_workers,
            epochs=bad_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            dev_num_batches=dev_num_batches,
            ckpt=ckpt,
            output_str=output_str,
            do_dev=False,
            log_freq=log_freq,
            get_tokenized=get_tokenized,
            overfit=overfit
        )
        print('Training on bad supp complete.\n')
    else: # let plotting proceed regardless
        bad_train_losses = []
        bad_dev_losses = []
    
    """ TRAINING - GOOD SUPP """
    if do_good:
        print('Training on good supp...')
        good_train_losses, good_dev_losses = train(
            loader_name='good_supp',
            tokenizers=tokenizers,
            batch_size=batch_size,
            num_batches=good_num_batches,
            max_length=max_length,
            lang_code=lang_code,
            num_workers=num_workers,
            epochs=good_epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            dev_num_batches=dev_num_batches,
            ckpt=ckpt,
            output_str=output_str,
            do_dev=False,
            log_freq=log_freq,
            get_tokenized=get_tokenized,
            overfit=overfit
        )
        print('Training on good supp complete.\n')
    else: # let plotting proceed regardless
        good_train_losses = []
        good_dev_losses = []
    
    """ TRAINING - TRAIN """
    print('Training on train...')
    train_train_losses, train_dev_losses = train(
        loader_name='train',
        tokenizers=tokenizers,
        batch_size=batch_size,
        num_batches=train_num_batches,
        max_length=max_length,
        lang_code=lang_code,
        num_workers=num_workers,
        epochs=train_epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dev_num_batches=dev_num_batches,
        ckpt=ckpt,
        output_str=output_str,
        do_dev=do_dev,
        log_freq=log_freq,
        get_tokenized=get_tokenized,
        overfit=overfit
    )
    print('Training on train complete.\n')
    
    # produce one train loss plot and one dev loss plot
    # splits are color-coded
    print('Plotting losses...')
    
    plot_losses(bad_train_losses, good_train_losses, train_train_losses, 'train', output_str)
    plot_losses(bad_dev_losses, good_dev_losses, train_dev_losses, 'dev', output_str)
    print(f'Done in {str(timedelta(seconds=time()-start))}\n')

if __name__ == '__main__':
    main()