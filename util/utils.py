import json
import torch
from torch import nn as nn
from torch.autograd import Variable
from library.dbengine import DBEngine
import numpy as np


def load_dataset(dataset_to_load):
    """Load the given dataset into memory."""
    print("Loading dataset : ", dataset_to_load)
    sql_data_path = 'data/tokenized_' + dataset_to_load + '.jsonl'
    table_data_path = 'data/tokenized_' + dataset_to_load + '.tables.jsonl'
    db_file = 'data/' + dataset_to_load + '.db'

    sql_data = []
    table_data = {}
    with open(sql_data_path, encoding="utf-8") as lines:
        for line in lines:
            sql = json.loads(line.strip())
            sql_data.append(sql)

    # Build a mapping of the tables with table_id as the key
    with open(table_data_path, encoding="utf-8") as lines:
        for line in lines:
            tab = json.loads(line.strip())
            table_data[tab[u'id']] = tab

    return sql_data, table_data, db_file


def best_model_name():
    """Function returns the filename that stores the best model found during training"""
    best_aggregate_model = 'saved_model/seq2sql.agg_model'
    best_selection_model = 'saved_model/seq2sql.sel_model'
    best_condition_model = 'saved_model/seq2sql.cond_'
    return best_aggregate_model, best_selection_model, best_condition_model


def generate_batch_sequence(sql_data, table_data, idxes, start, end):
    """Function creates a batch of input data given starting and ending indices."""
    # A container is created for each component of the input
    question_sequence = []
    column_sequence = []
    number_of_columns = []
    answer_sequence = []
    query_sequence = []
    ground_truth_condition_sequence = []
    raw_data = []
    for i in range(start, end):
        sql = sql_data[idxes[i]]
        question_sequence.append(sql['tokenized_question'])
        column_sequence.append(table_data[sql['table_id']]['tokenized_header'])
        number_of_columns.append(len(table_data[sql['table_id']]['header']))
        answer_sequence.append((sql['sql']['agg'],
                        sql['sql']['sel'],
                        len(sql['sql']['conds']),
                        tuple(x[0] for x in sql['sql']['conds']),
                        tuple(x[1] for x in sql['sql']['conds'])))
        query_sequence.append(sql['tokenized_query'])
        ground_truth_condition_sequence.append(sql['sql']['conds'])
        raw_data.append((sql['question'], table_data[sql['table_id']]['header'], sql['query']))


    return question_sequence, column_sequence, number_of_columns, answer_sequence, query_sequence,\
        ground_truth_condition_sequence, raw_data


def generate_batch_query(sql_data, idxes, start, end):
    query_gt = []
    table_ids = []
    for i in range(start, end):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


def epoch_train(model, optimizer, batch_size, sql_data, table_data):
    model.train()
    perm = np.random.permutation(len(sql_data))
    cumulative_loss = 0.0
    start = 0
    while start < len(sql_data):
        end = start + batch_size if start + batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, ground_truth_cond_seq, raw_data = \
            generate_batch_sequence(sql_data, table_data, perm, start, end)
        ground_truth_where_seq = model.generate_ground_truth_where_seq(q_seq, col_seq, query_seq)
        ground_truth_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, ground_truth_where=ground_truth_where_seq, ground_truth_cond=ground_truth_cond_seq, ground_truth_sel=ground_truth_sel_seq)
        loss = model.loss(score, ans_seq, ground_truth_where_seq)
        cumulative_loss += loss.data.cpu().numpy() * (end - start)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        start = end

    return cumulative_loss / len(sql_data)


def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    start = 0
    while start < len(sql_data):
        end = start + batch_size if start + batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, ground_truth_cond_seq, raw_data = \
            generate_batch_sequence(sql_data, table_data, perm, start, end)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = generate_batch_query(sql_data, perm, start, end)
        ground_truth_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, ground_truth_sel=ground_truth_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq)

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                                          sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)

        start = end

    return tot_acc_num / len(sql_data)


def epoch_acc(model, batch_size, sql_data, table_data, save_results = False):
    model.eval()
    perm = list(range(len(sql_data)))
    start = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while start < len(sql_data):
        end = start + batch_size if start + batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, ground_truth_cond_seq, raw_data =\
            generate_batch_sequence(sql_data, table_data, perm, start, end)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = generate_batch_query(sql_data, perm, start, end)
        ground_truth_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, ground_truth_sel=ground_truth_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                                       raw_q_seq, raw_col_seq)
        one_err, tot_err = model.check_accuracy(pred_queries, query_gt)
        
        if save_results:
            model.save_readable_results(pred_queries, query_gt, table_ids, table_data)

        one_acc_num += (end - start - one_err)
        tot_acc_num += (end - start - tot_err)

        start = end
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)


def load_word_embeddings(file_name):
    print('Loading word embedding from %s' % file_name)
    ret = {}
    with open(file_name, encoding="utf-8") as inf:
        for idx, line in enumerate(inf):
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array([float(x) for x in info[1:]])
    return ret


def run_lstm(lstm, inp, inp_len, hidden=None):
    sort_perm = np.array(sorted(range(len(inp_len)), key=lambda k: inp_len[k], reverse=True))
    sort_inp_len = inp_len[sort_perm]
    sort_perm_inv = np.argsort(sort_perm)
    if inp.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda()
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm],
                                                 sort_inp_len, batch_first=True)
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(
        sort_ret_s, batch_first=True)[0][sort_perm_inv]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    # Encode the columns.
    # The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len - 1]
    ret = torch.FloatTensor(len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st + cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len
