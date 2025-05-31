import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import StrLabelConverter
from torchvision import models
from main import instantiate_from_config
import functools

def disable_train(self, mode=True):
    return self

class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers if layers else [3, 8, 15, 22]  # Default layers to use for perceptual loss
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(self.layers) + 1)])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, img1, img2):
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        loss = 0
        for f1, f2 in zip(features1, features2):
            loss += F.mse_loss(f1, f2)
        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features

class RSSTE(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 decoder_config,
                 alphabet = '/model/rs-ste/data/alphabet/en.txt',
                 ckpt_path = None,
                 max_text_len = 32,
                 n_embd = 768,
                 training_mode='synth_pair',
                 ):
        super().__init__()
        self.training_mode = training_mode
        self.conv = nn.Conv2d(in_channels=3, out_channels=n_embd, kernel_size=4, stride=4, bias=False)
        self.conv_o = nn.Conv2d(in_channels=n_embd, out_channels=3, kernel_size=1)
        self.masked_rec_embd = nn.Embedding(1, n_embd)
        self.masked_img_imbd = nn.Embedding(1, n_embd)
        self.str_converter = StrLabelConverter(alphabet, max_text_len, 0)
        self.str_embd = nn.Embedding(len(self.str_converter.alphabet) + 1, n_embd)
        self.perceptual_loss_fn = PerceptualLoss()
        self.init_decoder_from_config(decoder_config)
        self.init_transformer_from_config(transformer_config, ckpt_path)
    
    def init_decoder_from_config(self, config):
        model = instantiate_from_config(config)
        model.eval()
        model.train = disable_train
        self.decoder = model

    
    def init_transformer_from_config(self, config, ckpt_path):
        self.transformer = instantiate_from_config(config)
        if ckpt_path is not None:
            print(f"Transformer restored from {ckpt_path}")

            from collections import OrderedDict
            sd = torch.load(ckpt_path)['state_dict']
            keys = list(sd.keys())
            new_keys = OrderedDict()
            model_state_dict = self.state_dict()
            for k in keys:
                new_k = k
                if k.startswith("vqgan"):
                    new_k = new_k.replace("vqgan","decoder")
                if new_k in model_state_dict:
                    new_keys[new_k] = sd[k]
            self.load_state_dict(new_keys,strict=True)
                    
    
    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        
        x = self.conv(x)
        return x # [bs, n_embd, 8, 32]

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        img1_latent = self.get_input('image1', batch).flatten(2).permute(0, 2, 1)
        rec2_indices, _ = self.str_converter.encode(batch['rec2'])
        rec2_indices = rec2_indices.to(self.device)
        rec2_embd = self.str_embd(rec2_indices)
        rec1_mask = self.masked_rec_embd.weight[0][None, None, :].expand(rec2_indices.shape[0], rec2_indices.shape[1], -1)
        img2_mask = self.masked_img_imbd.weight[0][None, None, :].expand(img1_latent.shape[0], img1_latent.shape[1], -1)

        inputs = torch.cat([rec2_embd, img1_latent, rec1_mask, img2_mask], dim=1)
        embeddings, logits = self.transformer(inputs)
        rec1_indices = torch.topk(logits[:,288:320,:], k=1, dim=-1)[1].view(batch["image1"].shape[0],-1)
        pred_rec = self.str_converter.decode(rec1_indices)
        img2_pred_latent = self.conv_o(embeddings[:,320:576,:].permute(0,2,1).contiguous().view(embeddings.shape[0], -1, 8, 32))
        img2_pred = self.decoder.decode(img2_pred_latent)
        img1 = batch['image1'].permute(0,3,1,2).to(memory_format=torch.contiguous_format)
        log['edited'] = img2_pred
        log["original"] = img1
        self.logger.experiment.add_text("train/pred_text", str(pred_rec), global_step = self.global_step)
        return log
    
    def forward(self, batch):
        output = dict()
        if self.training_mode == "synth_pair":
            img1_latent = self.get_input('image1', batch).flatten(2).permute(0, 2, 1)
            

            rec1_indices, _ = self.str_converter.encode(batch['rec1'])
            rec2_indices, _ = self.str_converter.encode(batch['rec2'])
            rec1_indices = rec1_indices.to(self.device)
            rec2_indices = rec2_indices.to(self.device)

            rec2_embd = self.str_embd(rec2_indices)
            rec1_mask = self.masked_rec_embd.weight[0][None, None, :].expand(rec1_indices.shape[0], rec1_indices.shape[1], -1)
            img2_mask = self.masked_img_imbd.weight[0][None, None, :].expand(img1_latent.shape[0], img1_latent.shape[1], -1)

            inputs = torch.cat([rec2_embd, img1_latent, rec1_mask, img2_mask], dim=1)

            embeddings, logits = self.transformer(inputs)

            img2_pred_latent = self.conv_o(embeddings[:,320:576,:].permute(0,2,1).contiguous().view(embeddings.shape[0], -1, 8, 32))
            img2_pred = self.decoder.decode(img2_pred_latent)
            img2 = batch['image2'].permute(0,3,1,2).to(memory_format=torch.contiguous_format)

            output['img2_pred'] = img2_pred
            output['logits'] = logits
            output['img2'] = img2
            output['rec1_indices'] = rec1_indices
            return output
        elif self.training_mode == "real_cycle":
            img1_latent = self.get_input('image1', batch).flatten(2).permute(0, 2, 1)
            
            rec1_indices, _ = self.str_converter.encode(batch['rec1'])
            rec2_indices, _ = self.str_converter.encode(batch['rec2'])
            rec1_indices = rec1_indices.to(self.device)
            rec2_indices = rec2_indices.to(self.device)

            rec1_embd = self.str_embd(rec1_indices)
            rec2_embd = self.str_embd(rec2_indices)

            rec1_mask = self.masked_rec_embd.weight[0][None, None, :].expand(rec1_indices.shape[0], rec1_indices.shape[1], -1)
            img2_mask = self.masked_img_imbd.weight[0][None, None, :].expand(img1_latent.shape[0], img1_latent.shape[1], -1)

            inputs = torch.cat([rec2_embd, img1_latent, rec1_mask, img2_mask], dim=1)
            embeddings1, logits1 = self.transformer(inputs)

            img2_pred_latent = self.conv_o(embeddings1[:,320:576,:].permute(0,2,1).contiguous().view(embeddings1.shape[0], -1, 8, 32))
            img2_pred = self.decoder.decode(img2_pred_latent)
            img2_pred_latent = self.conv(img2_pred).flatten(2).permute(0, 2, 1)
            inputs2 = torch.cat([rec1_embd, img2_pred_latent, rec1_mask, img2_mask], dim=1)
            embeddings2, logits2 = self.transformer(inputs2)

            img1_pred_latent = self.conv_o(embeddings2[:,320:576,:].permute(0,2,1).contiguous().view(embeddings2.shape[0], -1, 8, 32))
            img1_pred = self.decoder.decode(img1_pred_latent)


            img1 = batch['image1'].permute(0,3,1,2).to(memory_format=torch.contiguous_format)

            output['img1_pred'] = img1_pred
            output['img1_target'] = img1
            output['logits1'] = logits1
            output['logits2'] = logits2
            output['rec2_indices'] = rec2_indices
            output['rec1_indices'] = rec1_indices
            return output
    
    def training_step(self, batch, batch_idx):
        output = self(batch)
        if self.training_mode == "synth_pair":
            img_mse_loss = F.mse_loss(output['img2_pred'], output['img2'])
            rec_loss = F.cross_entropy(output["logits"][:,288:320,:].contiguous().view(-1, output["logits"][:,287:319,:].shape[-1]), output["rec1_indices"].view(-1))
            img_perceptual_loss = self.perceptual_loss_fn(output["img2_pred"], output["img2"])
            loss = 10*img_mse_loss + rec_loss + img_perceptual_loss
            self.log("train/img_mse_loss", img_mse_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/rec_loss", rec_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/img_perceptual_loss", img_perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss
        elif self.training_mode == "real_cycle":
            img_mse_loss = F.mse_loss(output['img1_pred'], output['img1_target'])
            img_perceptual_loss = self.perceptual_loss_fn(output['img1_pred'], output['img1_target'])
            rec1_loss = F.cross_entropy(output["logits1"][:,288:320,:].contiguous().view(-1, output["logits1"][:,288:320,:].shape[-1]), output["rec1_indices"].view(-1))
            rec2_loss = F.cross_entropy(output["logits2"][:,288:320,:].contiguous().view(-1, output["logits2"][:,288:320,:].shape[-1]), output["rec2_indices"].view(-1))
            loss = 10*img_mse_loss + img_perceptual_loss + 50*rec1_loss + 50*rec2_loss

            self.log("train/img_mse_loss", img_mse_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/img_perceptual_loss", img_perceptual_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/rec1_loss", rec1_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/rec2_loss", rec2_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            return loss
    
    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)
    
    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.LeakyReLU, torch.nn.BatchNorm2d)
        param_dict = {}
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn 
                if not fpn.startswith("decoder") and not fpn.startswith("perceptual_loss_fn"):
                    if pn.endswith('bias'):
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)
                    param_dict[fpn] = p

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        for param in self.decoder.parameters():
            param.requires_grad = False
        
        for param in self.perceptual_loss_fn.parameters():
            param.requires_grad = False
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
