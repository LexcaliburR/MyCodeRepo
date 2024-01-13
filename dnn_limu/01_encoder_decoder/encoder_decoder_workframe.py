import torch


class Encoder(nn.module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Init Encoder")

    def forward(self, x):
        pass

    
class Decoder(nn.module):
    def __init__(self, **kwargs):
        self.super().__init__()

    def _init_state(self, x):
        pass
    
    def forward(self, x):
        pass


class EncoderDecoder(nn.module):
    def __init__(self, encoder, decoder, **kwargs):
        self.super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, enc_input, dec_input):
        state = self.encoder(enc_input)
        self.decoder._init_state(state)