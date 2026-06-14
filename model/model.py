import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.nn import LSTM, Linear


class BiEAF(nn.Module):
    def __init__(self, args, pretrained):
        super(BiEAF, self).__init__()
        self.args = args

        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(args.char_vocab_size, args.char_dim, padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        self.char_conv = nn.Sequential(
            nn.Conv2d(1, args.char_channel_size, (args.char_dim, args.char_channel_width)),
            nn.ReLU()
        )

        # 2. Word Embedding Layer
        self.word_emb = nn.Embedding.from_pretrained(pretrained, freeze=True)

        # Highway network
        assert self.args.hidden_size * 2 == (self.args.char_channel_size + self.args.word_dim)
        for i in range(2):
            setattr(self, 'highway_linear{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2), nn.ReLU()))
            setattr(self, 'highway_gate{}'.format(i),
                    nn.Sequential(Linear(args.hidden_size * 2, args.hidden_size * 2), nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=args.hidden_size * 2,
                                 hidden_size=args.hidden_size,
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=args.dropout)

        # Self-attention weights: W_c and W_q (scalar importance score per position)
        self.self_att_c = Linear(args.hidden_size * 2, 1)
        self.self_att_q = Linear(args.hidden_size * 2, 1)

        # 5. Modeling Layer: f_c is hidden*2 (not hidden*8 as in BiDAF)
        self.modeling_LSTM1 = LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)
        self.modeling_LSTM2 = LSTM(input_size=args.hidden_size * 2,
                                   hidden_size=args.hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        # 6. Output Layer
        self.p1_weight = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.p2_weight = Linear(args.hidden_size * 2, 1, dropout=args.dropout)
        self.output_LSTM = LSTM(input_size=args.hidden_size * 2,
                                hidden_size=args.hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, batch):
        def char_emb_layer(x):
            """(batch, seq_len, word_len) -> (batch, seq_len, char_channel_size)"""
            batch_size = x.size(0)
            x = self.dropout(self.char_emb(x))
            x = x.transpose(2, 3)
            x = x.view(-1, self.args.char_dim, x.size(3)).unsqueeze(1)
            x = self.char_conv(x).squeeze()
            x = F.max_pool1d(x, x.size(2)).squeeze()
            x = x.view(batch_size, -1, self.args.char_channel_size)
            return x

        def highway_network(x1, x2):
            """(batch, seq_len, char_channel_size+word_dim) -> (batch, seq_len, hidden*2)"""
            x = torch.cat([x1, x2], dim=-1)
            for i in range(2):
                h = getattr(self, 'highway_linear{}'.format(i))(x)
                g = getattr(self, 'highway_gate{}'.format(i))(x)
                x = g * h + (1 - g) * x
            return x

        def self_att_layer(h, att_weight):
            """
            Paper §III-B intra-sentence self-attention:
              α = σ(W · h + b)   scalar importance score per position
              s = softmax(α)     normalized weights over the sequence
              g = s * h          importance-weighted representation
            (Paper writes 'Σ α · s'; implemented as s * h to preserve sequence shape.)
            """
            alpha = F.relu(att_weight(h))   # (batch, seq_len, 1)
            s = F.softmax(alpha, dim=1)     # (batch, seq_len, 1)
            return s * h                    # (batch, seq_len, hidden*2)

        def eaf_layer(g_c, g_q):
            """
            Paper §III-B cross-attention (Enhanced Attention Flow):
              α'_c = σ(g_c, g_q)    dot-product similarity
              α'_c = softmax(α'_c)  normalize over query positions
              f_c  = Σ α'_c · g_q  attended query representation

            NOTE: the paper writes 'Σ α'_c · α'_c' — the second term is a typo;
            it must be g_q (the query representations being attended to).
            """
            # (batch, c_len, hidden*2) x (batch, hidden*2, q_len) -> (batch, c_len, q_len)
            alpha = torch.bmm(g_c, g_q.transpose(1, 2))
            alpha = F.relu(alpha)               # σ activation
            alpha = F.softmax(alpha, dim=2)     # normalize over q_len
            # Correct formula: f_c = Σ α'_c · g_q
            return torch.bmm(alpha, g_q)        # (batch, c_len, hidden*2)

        def output_layer(m, l):
            """P_start = softmax(W_s · G),  P_end = softmax(W_e · G')"""
            p1 = self.p1_weight(m).squeeze(-1)  # (batch, c_len)
            m2 = self.output_LSTM((m, l))[0]    # (batch, c_len, hidden*2)
            p2 = self.p2_weight(m2).squeeze(-1) # (batch, c_len)
            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(batch.c_char)
        q_char = char_emb_layer(batch.q_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(batch.c_word[0])
        q_word = self.word_emb(batch.q_word[0])
        c_lens = batch.c_word[1]
        q_lens = batch.q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)

        # 3. Contextual Embedding Layer: h_c, h_q = BiLSTM(x_c), BiLSTM(x_q)
        h_c = self.context_LSTM((c, c_lens))[0]  # (batch, c_len, hidden*2)
        h_q = self.context_LSTM((q, q_lens))[0]  # (batch, q_len, hidden*2)

        # 4. Enhanced Attention Flow Layer
        # (a) Intra-sentence self-attention
        g_c = self_att_layer(h_c, self.self_att_c)  # (batch, c_len, hidden*2)
        g_q = self_att_layer(h_q, self.self_att_q)  # (batch, q_len, hidden*2)
        # (b) Cross-attention: f_c = Σ α'_c · g_q
        f_c = eaf_layer(g_c, g_q)                   # (batch, c_len, hidden*2)

        # 5. Modeling Layer: G = BiLSTM_1(BiLSTM_2(f_c))
        m = self.modeling_LSTM2((self.modeling_LSTM1((f_c, c_lens))[0], c_lens))[0]

        # 6. Output Layer
        p1, p2 = output_layer(m, c_lens)

        return p1, p2
