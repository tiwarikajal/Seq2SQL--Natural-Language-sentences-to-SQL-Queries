import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from seq2sql.model.word_embedding import WordEmbedding
from seq2sql.model.aggregation_predictor import AggregationPredictor
from seq2sql.model.selection_predictor import SelectionClausePredictor
from seq2sql.model.condition_predictor import ConditionPredictor
from library.table import Table
from library.query import Query


class Seq2SQL(nn.Module):
    """
    Seq2Sql Model which is internally comprised of three individual models
    1. Aggregation predictor
    2. Selection Predictor
    3. Condition Predictor
    """
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2,
                 gpu=False):
        super(Seq2SQL, self).__init__()

        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth

        self.maximum_column_count = 45
        self.maximum_token_count = 200
        self.SQL_SYNTAX_TOKENS = [
            '<UNK>', '<END>', 'WHERE', 'AND',
            'EQL', 'GT', 'LT', '<BEG>'
        ]

        # Word embedding
        self.embed_layer = WordEmbedding(word_emb, N_word, gpu, self.SQL_SYNTAX_TOKENS, our_model=False)

        # Model for predicting aggregation clause
        self.agg_pred = AggregationPredictor(N_word, N_h, N_depth)

        # Model for predicting select columns
        self.sel_pred = SelectionClausePredictor(N_word, N_h, N_depth, self.maximum_token_count)

        # Model for predicting the conditions
        self.cond_pred = ConditionPredictor(N_word, N_h, N_depth, self.maximum_column_count, self.maximum_token_count, gpu)

        # Loss function
        self.CE = nn.CrossEntropyLoss()
        if gpu:
            self.cuda()

    def generate_ground_truth_where_seq(self, q, col, query):
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            connect_col = [tok for col_tok in cur_col for tok in col_tok + [',']]
            all_toks = self.SQL_SYNTAX_TOKENS + connect_col + [None] + cur_q + [None]
            cur_seq = [all_toks.index('<BEG>')]
            if 'WHERE' in cur_query:
                cur_where_query = cur_query[cur_query.index('WHERE'):]
                cur_seq = cur_seq + list(map(lambda tok: all_toks.index(tok)
                if tok in all_toks else 0, cur_where_query))
            cur_seq.append(all_toks.index('<END>'))
            ret_seq.append(cur_seq)
        return ret_seq

    def forward(self, q, col, col_num, ground_truth_where=None, ground_truth_cond=None, ground_truth_sel=None):
        x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
        batch = self.embed_layer.gen_col_batch(col)
        col_inp_var, col_name_len, col_len = batch

        agg_score = self.agg_pred(x_emb_var, x_len)

        sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

        cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, ground_truth_where, ground_truth_cond)

        return (agg_score, sel_score, cond_score)

    def loss(self, score, truth_num, ground_truth_where):
        agg_score, sel_score, cond_score = score
        loss = 0
        agg_truth = list(map(lambda x: x[0], truth_num))
        data = torch.from_numpy(np.array(agg_truth))
        if self.gpu:
            agg_truth_var = Variable(data.cuda())
        else:
            agg_truth_var = Variable(data)

        loss += self.CE(agg_score, agg_truth_var.long())

        sel_truth = list(map(lambda x: x[1], truth_num))
        data = torch.from_numpy(np.array(sel_truth))
        if self.gpu:
            sel_truth_var = Variable(data).cuda()
        else:
            sel_truth_var = Variable(data)

        loss += self.CE(sel_score, sel_truth_var.long())

        for b in range(len(ground_truth_where)):
            if self.gpu:
                cond_truth_var = Variable(torch.from_numpy(np.array(ground_truth_where[b][1:])).cuda())
            else:
                cond_truth_var = Variable(torch.from_numpy(np.array(ground_truth_where[b][1:])))
            cond_pred_score = cond_score[b, :len(ground_truth_where[b]) - 1]

            loss += (self.CE(
                cond_pred_score, cond_truth_var.long()) / len(ground_truth_where))

        return loss

    def check_accuracy(self, pred_queries, ground_truth_queries):
        tot_err = agg_err = sel_err = cond_err = cond_num_err = \
            cond_col_err = cond_op_err = cond_val_err = 0.0
        for b, (pred_qry, ground_truth_qry) in enumerate(zip(pred_queries, ground_truth_queries)):
            good = True

            agg_pred = pred_qry['agg']
            agg_gt = ground_truth_qry['agg']
            if agg_pred != agg_gt:
                agg_err += 1
                good = False

            sel_pred = pred_qry['sel']
            sel_gt = ground_truth_qry['sel']
            if sel_pred != sel_gt:
                sel_err += 1
                good = False

            cond_pred = pred_qry['conds']
            cond_gt = ground_truth_qry['conds']
            flag = True
            if len(cond_pred) != len(cond_gt):
                flag = False
                cond_num_err += 1

            if flag and set(
                    x[0] for x in cond_pred) != set(x[0] for x in cond_gt):
                flag = False
                cond_col_err += 1

            for idx in range(len(cond_pred)):
                if not flag:
                    break
                ground_truth_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                if flag and cond_gt[ground_truth_idx][1] != cond_pred[idx][1]:
                    flag = False
                    cond_op_err += 1

            for idx in range(len(cond_pred)):
                if not flag:
                    break
                ground_truth_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                if flag and str(cond_gt[ground_truth_idx][2]).lower() != \
                        str(cond_pred[idx][2]).lower():
                    flag = False
                    cond_val_err += 1

            if not flag:
                cond_err += 1
                good = False

            if not good:
                tot_err += 1

        return np.array((agg_err, sel_err, cond_err)), tot_err

    def gen_query(self, score, q, col, raw_q, raw_col, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']',
                       '``': '"', '\'\'': '"', '--': u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and \
                        (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        agg_score, sel_score, cond_score = score

        ret_queries = []
        B = len(cond_score)
        for b in range(B):
            cur_query = {}
            cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
            cur_query['conds'] = []
            all_toks = self.SQL_SYNTAX_TOKENS + \
                       [x for toks in col[b] for x in
                        toks + [',']] + [''] + q[b] + ['']
            cond_toks = []
            for where_score in cond_score[b].data.cpu().numpy():
                cond_tok = np.argmax(where_score)
                cond_val = all_toks[cond_tok]
                if cond_val == '<END>':
                    break
                cond_toks.append(cond_val)

            if verbose:
                print(cond_toks)
            if len(cond_toks) > 0:
                cond_toks = cond_toks[1:]
            st = 0
            while st < len(cond_toks):
                cur_cond = [None, None, None]
                ed = len(cond_toks) if 'AND' not in cond_toks[st:] \
                    else cond_toks[st:].index('AND') + st
                if 'EQL' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('EQL') + st
                    cur_cond[1] = 0
                elif 'GT' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('GT') + st
                    cur_cond[1] = 1
                elif 'LT' in cond_toks[st:ed]:
                    op = cond_toks[st:ed].index('LT') + st
                    cur_cond[1] = 2
                else:
                    op = st
                    cur_cond[1] = 0
                sel_col = cond_toks[st:op]
                to_idx = [x.lower() for x in raw_col[b]]
                pred_col = merge_tokens(sel_col, raw_q[b] + ' || ' + \
                                        ' || '.join(raw_col[b]))
                if pred_col in to_idx:
                    cur_cond[0] = to_idx.index(pred_col)
                else:
                    cur_cond[0] = 0
                cur_cond[2] = merge_tokens(cond_toks[op + 1:ed], raw_q[b])
                cur_query['conds'].append(cur_cond)
                st = ed + 1
            ret_queries.append(cur_query)

        return ret_queries


    def save_readable_results(self, predicted_query, ground_truth, table_ids, table_data):
        file = open("./target_model_results.txt", "a+", encoding="utf-8")
        for index in range(len(predicted_query)):
            predicted_query_object = Query.from_dict(predicted_query[index])
            ground_truth_query_object = Query.from_dict(ground_truth[index])
            table_id = table_ids[index]
            table_info = table_data[table_id]
            table = Table(table_id, table_info["header"], table_info["types"], table_info["rows"])
            
            file.write(table.query_str(ground_truth_query_object))
            file.write("\n")
            file.write(table.query_str(predicted_query_object))
            file.write("\n\n")
        file.close()
