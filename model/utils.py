import torch
import torch.nn as nn

class StrLabelConverter(object):
    def __init__(self, alphabet, max_text_len, start_id):
        self.max_text_len = max_text_len
        if alphabet.endswith(".txt"):
            with open(alphabet, "r", encoding="utf-8") as f:
                alphabet = f.read()
        else:
            raise NotImplementedError("Only .txt alphabet files are supported")
        
        self.alphabet = alphabet
        self.dict = {}
        self.start_id = start_id
        for i, char in enumerate(self.alphabet):
            self.dict[char] = i + self.start_id
        self.pad_id = self.start_id + len(self.alphabet)
    
    def encode(self, text):
        if isinstance(text, str):
            try:
                text = [self.dict[char] for char in text]
            except KeyError:
                new_text = ""
                for c in text:
                    if c in self.alphabet:
                        new_text+= c
                text = [self.dict[char] for char in new_text]
            length = min(len(text), self.max_text_len)

            text = text[:self.max_text_len]
            text = text + [self.pad_id] * (self.max_text_len - length)
            return text, length
        
        elif isinstance(text, list):
            rec = []
            length = []
            for t in text:
                t, l = self.encode(t)
                rec.append(t)
                length.append(l)

            return torch.tensor(rec), length
    
    def decode(self, t,  scires=None):
        if t.ndim == 1:
            try:
                str = ""
                for c in t:
                    if int(c) == self.pad_id:
                        break
                    str += self.alphabet[int(c) - self.start_id]
                return str
            except IndexError:
                return "###"
        elif t.ndim == 2:
            results = []
            for i in range(t.shape[0]):
                results.append(self.decode(t[i]))
            return results


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h