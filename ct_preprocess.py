# coding: utf-8
import bz2
import pickle
from pathlib import Path

import joblib
import torch
from torch import optim
from sklearn import model_selection

import train
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


MAX_MSG_LEN = 200


def get_src_words(cache, idx):
    msgs = cache['message']
    y = cache['y']
    src_words = ([t.text for t in msgs[i].subject]
                 + [t.text for t in msgs[i].body
                    if t.ent_slots.ent[2:] not in {'EndOfMsg', 'Greeting'}]
                 for i in idx if y[i].gate > 0.5)
    for s in src_words:
        yield [Constants.BOS_WORD] + s[:MAX_MSG_LEN] + [Constants.EOS_WORD]


def get_tgt_words(cache, idx):
    msgs = cache['message']
    y = cache['y']
    tgt_words = ([t.ent_slots.slot.value for t in msgs[i].subject]
                 + [t.ent_slots.slot.value for t in msgs[i].body
                    if t.ent_slots.ent[2:] not in {'EndOfMsg', 'Greeting'}]
                 for i in idx if y[i].gate > 0.5)

    for s in tgt_words:
        yield [Constants.BOS_WORD] + s[:MAX_MSG_LEN] + [Constants.EOS_WORD]


def generate_split_data(cache_path, i_split, trn, tst):
    cache_path = Path(cache_path).expanduser()
    with bz2.open(cache_path, 'rb') as fh:
        cache = pickle.load(fh)

    for i in list(trn) + list(tst):
        y = cache['y'][i]
        m = cache['message'][i]
        m.attach_token_properties(ent_slots=y.y_ent)

    trn_src_words = get_src_words(cache, trn)
    vld_src_words = get_src_words(cache, tst)

    trn_tgt_words = get_tgt_words(cache, trn)
    vld_tgt_words = get_tgt_words(cache, tst)

    src_word2idx, train_src_insts = docs_to_index_seq(trn_src_words,
                                                      min_word_count=5)
    tgt_word2idx, train_tgt_insts = docs_to_index_seq(trn_tgt_words,
                                                      min_word_count=5)

    valid_src_insts = [[src_word2idx.get(w, Constants.UNK) for w in s]
                       for s in vld_src_words]

    valid_tgt_insts = [[tgt_word2idx.get(w, Constants.UNK) for w in s]
                       for s in vld_tgt_words]

    data = {'settings': {'max_word_seq_len': MAX_MSG_LEN},
            'dict': {'src': src_word2idx,
                     'tgt': tgt_word2idx},
            'train': {'src': train_src_insts,
                      'tgt': train_tgt_insts},
            'valid': {'src': valid_src_insts,
                      'tgt': valid_tgt_insts}}

    out_fn = f'data/ct-inscope-{i_split:02d}.tok.pt'
    torch.save(data, out_fn)
    return out_fn


def docs_to_index_seq(docs, min_word_count=1, word2idx=None):
    ''' Trim vocab by number of occurence '''
    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    idx_seqs = []
    word_counts = [min_word_count] * len(word2idx)

    for sent in docs:
        idx = []
        for w in sent:
            if w not in word2idx:
                w_idx = len(word2idx)
                word2idx[w] = w_idx
                word_counts.append(1)
            else:
                w_idx = word2idx[w]
                word_counts[w_idx] += 1
            assert len(word_counts) == len(word2idx)

            idx.append(w_idx)
        idx_seqs.append(idx)

    idx_seqs = [[idx if word_counts[idx] >= min_word_count
                 else Constants.UNK for idx in s]
                for s in idx_seqs]

    return word2idx, idx_seqs


if __name__ == '__main__':
    N = 68669  # hard coded because loading the cache is a pain
    N = 5000  # hard coded because loading the cache is a pain
    cv = model_selection.KFold(n_splits=10)
    cache_path = '~/ct/ct-nlp-data/NLPPipeline_dsk_cache.pbz'
    jobs = ((cache_path, i_split, trn, tst)
            for i_split, (trn, tst) in enumerate(cv.split(list(range(N)))))
    parallel = joblib.Parallel(n_jobs=1)
    data_files = parallel(joblib.delayed(generate_split_data)(*job)
                          for job in jobs)

    for i_split, datafile in enumerate(data_files):
        data = torch.load(datafile)
        args = train.parser.parse_args()

        trn_data, val_data =\
            train.prepare_dataloaders(data, batch_size=args.batch_size)
        src_vocab_size = trn_data.dataset.src_vocab_size
        tgt_vocab_size = trn_data.dataset.tgt_vocab_size

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
