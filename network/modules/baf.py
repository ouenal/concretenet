# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.network_utils import mean_pooling


class BAFModule(nn.Module):
    def __init__(self, input_dim=16, # Dim of the candidate features
                 dim=128,            # Dim of model and input word embeddings
                 num_heads=4,        # Number of heads
                 num_layers=2,       # Number of layers before radius change
                 radii=[1.0, 2.5, 100.0]):
        super().__init__()
        self.radii = radii
        self.match = nn.Sequential(nn.Linear(input_dim, dim), nn.ReLU())
        self.fuse_visual = nn.Linear(len(self.radii)*dim, dim)
        self.classification_head = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Conv1d(dim, dim, 1),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Conv1d(dim, 1, 1)
        )

        self.num_heads = num_heads
        encoder_layer = nn.TransformerEncoderLayer(dim, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        decoder_layer = nn.TransformerDecoderLayer(dim, num_heads, batch_first=True)
        self.decoder = nn.ModuleList([
            nn.TransformerDecoder(decoder_layer, num_layers)
            for _ in range(len(self.radii))
        ])

        self.camera_embedding = nn.Embedding(1, dim)
        self.fuse_camera = nn.Sequential(
            nn.Linear(3*dim, dim),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Linear(dim, 3)
        )

    def forward(self, input_dict, visual_ret, verbal_ret):
        ret = {}
        padding_mask = verbal_ret['padding_mask']
        f_verbal = self.encoder(verbal_ret['f_word'], src_key_padding_mask=~padding_mask.bool())
        ret['f_sentence'] = F.normalize(mean_pooling(f_verbal, padding_mask), p=2, dim=1)

        f_visual = self.match(visual_ret['f_kernel'])[None].repeat(f_verbal.shape[0],1,1) # (L,I,C)

        batch_verbal = input_dict['sentence_batch']
        batch_visual = visual_ret['candidate_batch']
        batch_mask = torch.eq(batch_verbal[:,None], batch_visual[None])
        batch_mask[batch_mask.sum(1)==0] = 1 # Hot fix
        batch_mask = torch.eq(batch_mask[:,:,None], batch_mask[:,None])

        candidate_centers = visual_ret['candidate_centers']
        candidate_centers += batch_visual[:,None] * 1000 # Batch margin
        d_candidate = torch.norm(candidate_centers[:,None] - candidate_centers[None], p=2, dim=-1)

        gct = self.camera_embedding.weight[None].repeat(f_verbal.shape[0],1,1) # (L,1,128)
        query_tokens = torch.cat((f_visual, gct), dim=1)

        pyramid_f_visual = []
        pyramid_gct = []
        for decoder_layer, radius in zip(self.decoder, self.radii):
            spherical_mask = d_candidate < radius
            spherical_mask = spherical_mask[None].repeat(f_verbal.shape[0],1,1)
            spherical_mask = spherical_mask & batch_mask
            # Pad spherical mask for GCT
            padded_spherical_mask = F.pad(spherical_mask, (0,1,0,1), "constant", True)
            padded_spherical_mask = padded_spherical_mask[:,None].repeat(1,self.num_heads,1,1).flatten(0,1)

            query_tokens = decoder_layer(query_tokens, f_verbal,
                                         tgt_mask=~padded_spherical_mask,
                                         memory_key_padding_mask=~padding_mask.bool())
            pyramid_f_visual.append(query_tokens[:,:-1])
            pyramid_gct.append(query_tokens[:,-1])
        f_visual = self.fuse_visual(torch.cat(pyramid_f_visual, dim=-1))
        cam_output = self.fuse_camera(torch.cat(pyramid_gct, dim=-1))
        lang_output = self.classification_head(f_visual.permute(0,2,1)).squeeze(1)
        ret.update({'f_instance': f_visual, 'cam_output': cam_output, 'lang_output': lang_output})
        return ret

    def decode_output(self, output_dict):
        return {'lang_output': output_dict['lang_output']}
