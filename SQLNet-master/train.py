import torch
from sqlnet.utils import *
from sqlnet.model.seq2sql import Seq2SQL
import datetime
import xml.etree.ElementTree as ET
from lxml import etree
etree.XMLParser().
ET.fromstring
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true',
                        help='If set, use small data; used for fast debugging.')
    args = parser.parse_args()

    N_word = 300
    B_word = 6
    if args.toy:
        USE_SMALL = True
        GPU = True
        BATCH_SIZE = 15
    else:
        USE_SMALL = False
        GPU = True
        BATCH_SIZE = 64

    learning_rate = 1e-3

    sql_data, table_data, val_sql_data, val_table_data, test_sql_data, test_table_data, \
        TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)

    TRAIN_ENTRY = (True, True, True)  # (AGG, SEL, COND)
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), use_small=USE_SMALL)

    model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU)
    optimizer = torch.optim.Adam(
        model.parameters(), lr = learning_rate, weight_decay = 0)

    agg_m, sel_m, cond_m = best_model_name(args)

    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
    best_agg_acc = init_acc[1][0]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][1]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][2]
    best_cond_idx = 0
    print('Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s' % init_acc)

    if TRAIN_AGG:
        torch.save(model.agg_pred.state_dict(), agg_m)
    if TRAIN_SEL:
        torch.save(model.sel_pred.state_dict(), sel_m)
    if TRAIN_COND:
        torch.save(model.cond_pred.state_dict(), cond_m)
    for i in range(50):
        print('Epoch %d @ %s' % (i + 1, datetime.datetime.now()))
        print(' Loss = %s' % epoch_train(
            model, optimizer, BATCH_SIZE,
            sql_data, table_data, TRAIN_ENTRY))
        print(' Train acc_qm: %s\n   breakdown result: %s' % epoch_acc(
            model, BATCH_SIZE, sql_data, table_data, TRAIN_ENTRY))
        val_acc = epoch_acc(model,
                            BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY)
        print(' Dev acc_qm: %s\n   breakdown result: %s' % val_acc)
        if TRAIN_AGG:
            if val_acc[1][0] > best_agg_acc:
                best_agg_acc = val_acc[1][0]
                best_agg_idx = i + 1
                torch.save(model.agg_pred.state_dict(),
                           'saved_model/epoch%d.agg_model' % (i + 1))
                torch.save(model.agg_pred.state_dict(), agg_m)
        if TRAIN_SEL:
            if val_acc[1][1] > best_sel_acc:
                best_sel_acc = val_acc[1][1]
                best_sel_idx = i + 1
                torch.save(model.sel_pred.state_dict(),
                           'saved_model/epoch%d.sel_model' % (i + 1))
                torch.save(model.sel_pred.state_dict(), sel_m)
        if TRAIN_COND:
            if val_acc[1][2] > best_cond_acc:
                best_cond_acc = val_acc[1][2]
                best_cond_idx = i + 1
                torch.save(model.cond_pred.state_dict(),
                           'saved_model/epoch%d.cond_model' % (i + 1))
                torch.save(model.cond_pred.state_dict(), cond_m)
        print(' Best val acc = %s, on epoch %s individually' % (
            (best_agg_acc, best_sel_acc, best_cond_acc),
            (best_agg_idx, best_sel_idx, best_cond_idx)))
