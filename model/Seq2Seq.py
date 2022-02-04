import torch
import torch.nn

class Seq2Seq(nn.Module):
    def __init(self,
               encoder:None,
               decoder:None,
               src_pad_idx,
               trg_pad_idx,):
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self,src):
        src_mask = (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self,trg):
        trg_pad_mask = (trg!=self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril((trg_len, trg_len)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forword(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        encoder_output = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, encoder_output, trg_mask, src_mask)

        return output, attention