# coding: utf-8
import bz2
import pickle
from pathlib import Path

import torch
from torch import optim
from sklearn import model_selection

import train
import preprocess
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


cache_path = Path('~/ct/ct-nlp-data/NLPPipeline_dsk_cache.pbz').expanduser()
with bz2.open(cache_path, 'rb') as fh:
    cache = pickle.load(fh)


for i, m in enumerate(cache['message']):
    m.attach_token_properties(ent_slots=cache['y'][i].y_ent)


MAX_MSG_LEN = 200


def get_src_words(cache, idx):
    msgs = cache['message']
    src_words = [[Constants.BOS_WORD]
                 + [t.text for t in msgs[i].subject]
                 + [t.text for t in msgs[i].body
                    if t.ent_slots.ent[2:] not in {'EndOfMsg', 'Greeting'}]
                 + [Constants.EOS_WORD]
                 for i in idx]
    src_words = [words[:MAX_MSG_LEN + 2] for words in src_words]
    return src_words


def get_tgt_words(cache, idx):
    msgs = cache['message']
    tgt_words = [[Constants.BOS_WORD]
                 + [t.ent_slots.slot.value for t in msgs[i].subject]
                 + [t.ent_slots.slot.value for t in msgs[i].body
                    if t.ent_slots.ent[2:] not in {'EndOfMsg', 'Greeting'}]
                 + [Constants.EOS_WORD]
                 for i in idx]
    tgt_words = [words[:MAX_MSG_LEN + 2] for words in tgt_words]
    return tgt_words


def generate_split_data(cache, trn, tst):
    trn_src_words = get_src_words(cache, trn)
    vld_src_words = get_src_words(cache, tst)

    trn_tgt_words = get_tgt_words(cache, trn)
    vld_tgt_words = get_tgt_words(cache, tst)

    src_word2idx = preprocess.build_vocab_idx(trn_src_words, min_word_count=5)
    tgt_word2idx = preprocess.build_vocab_idx(trn_tgt_words, min_word_count=5)

    train_src_insts = preprocess.convert_instance_to_idx_seq(trn_src_words,
                                                             src_word2idx)
    valid_src_insts = preprocess.convert_instance_to_idx_seq(vld_src_words,
                                                             src_word2idx)
    train_tgt_insts = preprocess.convert_instance_to_idx_seq(trn_tgt_words,
                                                             tgt_word2idx)
    valid_tgt_insts = preprocess.convert_instance_to_idx_seq(vld_tgt_words,
                                                             tgt_word2idx)

    data = {'settings': {'max_word_seq_len': MAX_MSG_LEN},
            'dict': {'src': src_word2idx,
                     'tgt': tgt_word2idx},
            'train': {'src': train_src_insts[:128],
                      'tgt': train_tgt_insts[:128]},
            'valid': {'src': valid_src_insts[:128],
                      'tgt': valid_tgt_insts[:128]}}

    torch.save(data, out_fn)
    return out_fn


N = len(cache['message'])
cv = model_selection.KFold(n_splits=10)
for i_split, (trn, tst) in enumerate(cv.split(list(range(N)))):
    out_fn = f'ct_full-{i_split:02d}.tok.pt'
    # generate_split_data(cache, trn, tst)
    data = torch.load(f'ct_full-{i_split:02d}.tok.pt')
    args = train.parser.parse_args(['-data', ''])

    trn_data, val_data = train.prepare_dataloaders(data,
                                                   batch_size=16)
    src_vocab_size = trn_data.dataset.src_vocab_size
    tgt_vocab_size = trn_data.dataset.tgt_vocab_size

    DEVICE = 'cpu'
    mdl = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            # include <BOS> and <EOS>
            data['settings'].get('max_word_seq_len', 200) + 2,
            tgt_emb_prj_weight_sharing=args.proj_share_weight,
            emb_src_tgt_weight_sharing=args.embs_share_weight,
            d_k=args.d_k,
            d_v=args.d_v,
            d_model=args.d_model,
            d_word_vec=args.d_word_vec,
            d_inner=args.d_inner_hid,
            n_layers=args.n_layers,
            n_head=args.n_head,
            dropout=args.dropout).to(DEVICE)

    optimizer = ScheduledOptim(
            optim.Adam(filter(lambda x: x.requires_grad, mdl.parameters()),
                       betas=(0.9, 0.98), eps=1e-09),
            args.d_model, args.n_warmup_steps)

    train.train(mdl, trn_data, val_data, optimizer, DEVICE, args)
