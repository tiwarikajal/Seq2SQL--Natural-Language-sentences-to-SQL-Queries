from util.utils import *
from seq2sql.model.seq2sql import Seq2SQL
import datetime
import argparse
from util.constants import *
from util.graph_plotter import plot_data


def train_seq2sql():
    
    USE_SMALL = True
    
    learning_rate = 1e-3

    sql_data, table_data, val_sql_data, val_table_data, test_sql_data, test_table_data, \
    TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.6B.300d.txt', use_small=USE_SMALL)

    model = Seq2SQL(word_emb, N_word=300, gpu=GPU)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    agg_m, sel_m, cond_m = best_model_name()

    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data)
    best_agg_acc = init_acc[1][0]
    best_agg_idx = 0
    best_sel_acc = init_acc[1][1]
    best_sel_idx = 0
    best_cond_acc = init_acc[1][2]
    best_cond_idx = 0
    print('Init dev acc_qm: %s\n  breakdown on (agg, sel, where): %s' % init_acc)

    torch.save(model.sel_pred.state_dict(), sel_m)
    torch.save(model.agg_pred.state_dict(), agg_m)
    torch.save(model.cond_pred.state_dict(), cond_m)
    epoch_losses = []

    for i in range(TRAINING_EPOCHS):
        print('Epoch %d @ %s' % (i + 1, datetime.datetime.now()))
        epoch_loss = epoch_train(model, optimizer, BATCH_SIZE, sql_data, table_data)
        epoch_losses.append(epoch_loss)
        print(' Loss = %s' % epoch_loss)
        print(' Train acc_qm: %s\n   breakdown result: %s' % epoch_acc(model, BATCH_SIZE, sql_data, table_data))
        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data)
        print(' Dev acc_qm: %s\n   breakdown result: %s' % val_acc)
        if val_acc[1][0] > best_agg_acc:
            best_agg_acc = val_acc[1][0]
            best_agg_idx = i + 1
            torch.save(model.agg_pred.state_dict(), 'saved_model/epoch%d.agg_model' % (i + 1))
            torch.save(model.agg_pred.state_dict(), agg_m)
        if val_acc[1][1] > best_sel_acc:
            best_sel_acc = val_acc[1][1]
            best_sel_idx = i + 1
            torch.save(model.sel_pred.state_dict(), 'saved_model/epoch%d.sel_model' % (i + 1))
            torch.save(model.sel_pred.state_dict(), sel_m)
        if val_acc[1][2] > best_cond_acc:
            best_cond_acc = val_acc[1][2]
            best_cond_idx = i + 1
            torch.save(model.cond_pred.state_dict(), 'saved_model/epoch%d.cond_model' % (i + 1))
            torch.save(model.cond_pred.state_dict(), cond_m)

        print(' Best val acc = %s, on epoch %s individually' % (
            (best_agg_acc, best_sel_acc, best_cond_acc),
            (best_agg_idx, best_sel_idx, best_cond_idx)))

    plot_data(x = range(TRAINING_EPOCHS), y = epoch_losses, xlabel = "Epochs", ylabel = "Loss", label = "Loss Graph for target seq2sql model")


if __name__ == '__main__':
    train_seq2sql()
