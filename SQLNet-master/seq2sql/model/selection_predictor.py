import torch.nn as nn
from util.utils import run_lstm, col_name_encode


class SelectionClausePredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num):
        super(SelectionClausePredictor, self).__init__()
        self.max_tok_num = max_tok_num
        self.sel_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h // 2, num_layers=N_depth, batch_first=True,
                                dropout=0.3, bidirectional=True)
        self.sel_att = nn.Linear(N_h, 1)
        self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h // 2,
                                        num_layers=N_depth, batch_first=True,
                                        dropout=0.3, bidirectional=True)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax()

    def forward(self, x_emb_var, x_len, col_inp_var,
                col_name_len, col_len, col_num):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        e_col, _ = col_name_encode(col_inp_var, col_name_len,
                                   col_len, self.sel_col_name_enc)

        h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
        att_val = self.sel_att(h_enc).squeeze()
        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val)
        K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
        K_sel_expand = K_sel.unsqueeze(1)

        sel_score = self.sel_out(self.sel_out_K(K_sel_expand) + \
                                 self.sel_out_col(e_col)).squeeze()
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100

        return sel_score
