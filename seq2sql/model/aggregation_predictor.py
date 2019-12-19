import torch.nn as nn
from util.utils import run_lstm


class AggregationPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth):
        super(AggregationPredictor, self).__init__()

        self.agg_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h // 2,
                                num_layers=N_depth, batch_first=True,
                                dropout=0.3, bidirectional=True)

        self.agg_att = nn.Linear(N_h, 1)
        self.agg_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(), nn.Linear(N_h, 6))
        self.softmax = nn.Softmax()

    def forward(self, x_emb_var, x_len, col_inp_var=None, col_name_len=None, col_len=None, col_num=None, ground_truth_sel=None):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)
        att_val = self.agg_att(h_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val)

        K_agg = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
        agg_score = self.agg_out(K_agg)
        return agg_score
