from util.utils import *
from seq2sql.model.seq2sql import Seq2SQL
import argparse


def test_seq2sql():
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
        USE_SMALL = True
        GPU = True
        BATCH_SIZE = 64

    sql_data, table_data, val_sql_data, val_table_data, \
    test_sql_data, test_table_data, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt' % (B_word, N_word), use_small=USE_SMALL)
    model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU)

    agg_m, sel_m, cond_m = best_model_name()
    print("Loading from %s" % agg_m)
    model.agg_pred.load_state_dict(torch.load(agg_m))
    print("Loading from %s" % sel_m)
    model.sel_pred.load_state_dict(torch.load(sel_m))
    print("Loading from %s" % cond_m)
    model.cond_pred.load_state_dict(torch.load(cond_m))

    print("Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s" % epoch_acc(model, BATCH_SIZE, val_sql_data,
                                                                              val_table_data))
    print("Dev execution acc: %s" % epoch_exec_acc(
        model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB))
    print("Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s" % epoch_acc(model, BATCH_SIZE, test_sql_data,
                                                                               test_table_data))
    print("Test execution acc: %s" % epoch_exec_acc(
        model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB))


if __name__ == '__main__':
    test_seq2sql()
