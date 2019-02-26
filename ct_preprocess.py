# coding: utf-8
import bz2
import pickle
from pathlib import Path

import torch
from torch import optim

import train
import preprocess
import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim


with bz2.open(Path('~/ct/ct-nlp-data/NLPPipeline_dsk_cache_SAMPLE.pbz').expanduser(), 'rb') as fh:
    cache = pickle.load(fh)

for i, m in enumerate(cache['message']):
   m.attach_token_properties(ent_slots=cache['y'][i].y_ent) 

MAX_MSG_LEN = 200

trn_src_words = [[Constants.BOS_WORD]
                 + [t.text for t in m.subject] 
                 + [t.text for t in m.body if t.ent_slots.ent not in {'B-EndOfMsg', 'I-EndOfMsg'}] 
                 + [Constants.EOS_WORD]
                for m in cache['message'][:-674]]
trn_src_words = [words for words in trn_src_words if len(words) < MAX_MSG_LEN + 2]

vld_src_words = [[Constants.BOS_WORD]
                 + [t.text for t in m.subject]
                 + [t.text for t in m.body if t.ent_slots.ent not in {'B-EndOfMsg', 'I-EndOfMsg'}]
                 + [Constants.EOS_WORD]
                for m in cache['message'][-674:]]
vld_src_words = [words for words in vld_src_words if len(words) < MAX_MSG_LEN + 2]

trn_tgt_words = [[Constants.BOS_WORD]
                 + [t.ent_slots.slot.value for t in m.subject]
                 + [t.ent_slots.slot.value for t in m.body if t.ent_slots.ent not in {'B-EndOfMsg', 'I-EndOfMsg'}]
                 + [Constants.EOS_WORD]
                for m in cache['message'][:-674]]
trn_tgt_words = [words for words in trn_tgt_words if len(words) < MAX_MSG_LEN + 2]

vld_tgt_words = [[Constants.BOS_WORD]
                 + [t.ent_slots.slot.value for t in m.subject]
                 + [t.ent_slots.slot.value for t in m.body if t.ent_slots.ent not in {'B-EndOfMsg', 'I-EndOfMsg'}]
                 + [Constants.EOS_WORD]
                for m in cache['message'][-674:]]
vld_tgt_words = [words for words in vld_tgt_words if len(words) < MAX_MSG_LEN + 2]

src_word2idx = preprocess.build_vocab_idx(trn_src_words, min_word_count=5)
tgt_word2idx = preprocess.build_vocab_idx(trn_tgt_words, min_word_count=5)
train_src_insts = preprocess.convert_instance_to_idx_seq(trn_src_words, src_word2idx)
valid_src_insts = preprocess.convert_instance_to_idx_seq(vld_src_words, src_word2idx)
train_tgt_insts = preprocess.convert_instance_to_idx_seq(trn_tgt_words, tgt_word2idx)
valid_tgt_insts = preprocess.convert_instance_to_idx_seq(vld_tgt_words, tgt_word2idx)

data = {
    'settings': {'max_word_seq_len': MAX_MSG_LEN},
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

torch.save(data, 'ct_sample.tok.pt')
data = torch.load('ct_sample.tok.pt')

training_data, validation_data = train.prepare_dataloaders(data, batch_size=64)
args = train.parser.parse_args(['-data', ''])
src_vocab_size = training_data.dataset.src_vocab_size
tgt_vocab_size = training_data.dataset.tgt_vocab_size

DEVICE = 'cpu'
mdl = Transformer(
        src_vocab_size,
        tgt_vocab_size,
        data['settings'].get('max_word_seq_len', 200), 
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
        optim.Adam(
            filter(lambda x: x.requires_grad, mdl.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
            args.d_model, args.n_warmup_steps)

train.train(mdl, training_data, validation_data, optimizer, DEVICE, args)

