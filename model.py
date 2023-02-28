from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from utils import Tensor
from utils import assert_shape
from utils import build_grid
from utils import conv_transpose_out_shape

from torch import linalg as LA

from params import SAViParams
params = SAViParams()



class Corrector(nn.Module):
    def __init__(self, in_features, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size  # number of hidden layers in slot dimensions
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        num_heads = 4

        self.norm_inputs = nn.LayerNorm(self.in_features)
        # I guess this is layer norm across each slot? should look into this
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_slots_after_cat = nn.LayerNorm(self.slot_size*2)
        self.norm_mlp = nn.LayerNorm(self.slot_size)
        
        self.multihead_attn = nn.MultiheadAttention(self.slot_size, num_heads)
        
        self.mlp_attn = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k1 = nn.Linear(self.in_features, self.slot_size, bias=False)
        self.project_v1 = nn.Linear(self.in_features, self.slot_size, bias=False)
       

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)


        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )


        self.register_buffer(
            "slots_mu",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )
        self.register_buffer(
            "slots_log_sigma",
            nn.init.xavier_uniform_(torch.zeros((1, 1, self.slot_size)), gain=nn.init.calculate_gain("linear")),
        )


    def forward(self, inputs: Tensor, instant_slots, i):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        
        k1 = self.project_k1(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(k1.size(), (batch_size, num_inputs, self.slot_size))
        v1 = self.project_v1(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        assert_shape(v1.size(), (batch_size, num_inputs, self.slot_size))

        instant_slots = self.norm_slots(instant_slots)        
        slots_prev = instant_slots
        
        
        if i != 1:
            instant_slots = instant_slots.permute(1, 0, 2)
            instant_slots, _ = self.multihead_attn(instant_slots, instant_slots, instant_slots)
            instant_slots = instant_slots.permute(1, 0, 2)
            instant_slots = slots_prev + instant_slots
            instant_slots = self.norm_slots(instant_slots)
            instant_slots = self.mlp_attn(instant_slots) + instant_slots

        
        
        for _ in range(self.num_iterations):
            slots_prev = instant_slots
            instant_slots = self.norm_slots(instant_slots)

            # Attention.
            q = self.project_q(instant_slots)  # Shape: [batch_size, num_slots, slot_size].
            assert_shape(q.size(), (batch_size, self.num_slots, self.slot_size))

            attn_norm_factor = self.slot_size ** -0.5
            attn_logits1 = attn_norm_factor * torch.matmul(k1, q.transpose(2, 1))
            attn1 = F.softmax(attn_logits1, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].
            assert_shape(attn1.size(), (batch_size, num_inputs, self.num_slots))

            # Weighted mean.
            attn1 = attn1 + self.epsilon
            attn1 = attn1 / torch.sum(attn1, dim=1, keepdim=True)
            updates = torch.matmul(attn1.transpose(1, 2), v1)
            # `updates` has shape: [batch_size, num_slots, slot_size].
            assert_shape(updates.size(), (batch_size, self.num_slots, self.slot_size))

            # Slot update.
            # GRU is expecting inputs of size (N,H) so flatten batch and slots dimension
            instant_slots = self.gru(
                updates.view(batch_size * self.num_slots, self.slot_size),
                slots_prev.view(batch_size * self.num_slots, self.slot_size),
            )
            
            instant_slots = instant_slots.view(batch_size, self.num_slots, self.slot_size)
            assert_shape(instant_slots.size(), (batch_size, self.num_slots, self.slot_size))
            instant_slots = instant_slots + self.mlp(self.norm_mlp(instant_slots))
            assert_shape(instant_slots.size(), (batch_size, self.num_slots, self.slot_size))

        return instant_slots


class SAViModel(nn.Module):
    def __init__(
        self,
        resolution: Tuple[int, int],
        num_slots: int,
        num_iterations: int,
        in_channels: int = 3,
        kernel_size: int = 5,
        slot_size: int = 128,
        hidden_dims: Tuple[int, ...] = params.hidden_dims,
        decoder_hidden_dims: Tuple[int, ...] = params.decoder_hidden_dims,
        decoder_resolution: Tuple[int, int] = (8, 8),
        empty_cache=False,
    ):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.slot_size = slot_size
        self.empty_cache = empty_cache
        self.hidden_dims = hidden_dims
        self.decoder_hidden_dims = decoder_hidden_dims
        self.decoder_resolution = decoder_resolution
        self.out_features = self.hidden_dims[-1]
        self.slot_attn_in_features = self.out_features*2

        modules = []
        channels = self.in_channels
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        out_channels=h_dim,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 2,
                    ),
                    nn.ReLU(),
                )
            )
            channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.out_features, resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.Linear(self.out_features, self.slot_attn_in_features),
            nn.ReLU(),
            nn.Linear(self.slot_attn_in_features, self.slot_attn_in_features),
        )

        # Build Decoder
        modules = []

        in_size = decoder_resolution[0]
        out_size = in_size
        
        for i in range(len(self.decoder_hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.decoder_hidden_dims[i],
                        self.decoder_hidden_dims[i+1],
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    ),
                    nn.ReLU(),
                )
            )
            out_size = conv_transpose_out_shape(out_size, 2, 2, 5, 1)
            
        assert_shape(
            resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. Try changing `decoder_resolution`.",
        )

        # same convolutions
        modules.append(nn.ConvTranspose2d(self.decoder_hidden_dims[-1], 4, kernel_size=5, stride=1, padding=2, output_padding=0))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.in_channels, self.slot_size, self.decoder_resolution)
        

        self.corrector = Corrector(
            in_features=self.slot_attn_in_features,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=128,
        )

    def forward(self, batch):
        if self.empty_cache:
            torch.cuda.empty_cache()

        batch_size, nFrames, num_channels, height, width = batch.shape
        #print(batch.shape)
        outs = []
        slots_all = []

        slots = torch.randn((batch_size, self.num_slots, self.slot_size)).cuda()
        slots = self.corrector.slots_mu + self.corrector.slots_log_sigma.exp() * slots
        prev_slot = slots

        for i in range(nFrames):                        
        
            x = batch[:,i,:,:,:]
            encoder_out = self.encoder(x)
            encoder_out = self.encoder_pos_embedding(encoder_out)
            # `encoder_out` has shape: [batch_size, filter_size, height, width]
            encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
            # `encoder_out` has shape: [batch_size, filter_size, height*width]
            encoder_out = encoder_out.permute(0, 2, 1)
            encoder_out = self.encoder_out_layer(encoder_out)
            # `encoder_out` has shape: [batch_size, height*width, filter_size]
            

            slots = self.corrector(encoder_out, prev_slot, i)
            assert_shape(slots.size(), (batch_size, self.num_slots, self.slot_size))            

            prev_slot = slots
                
            # `slots` has shape: [batch_size, num_slots, slot_size].
            batch_size, num_slots, slot_size = slots.shape
            
            slots = slots.view(batch_size * num_slots, slot_size, 1, 1)
            slots_all.append(slots)
            decoder_in = slots.repeat(1, 1, self.decoder_resolution[0], self.decoder_resolution[1])

            out = self.decoder_pos_embedding(decoder_in)
            out = self.decoder(out)
            outs.append(out)
            
            
        out = torch.stack(outs, dim = 1)
        slots_all = torch.stack(slots_all, dim = 1)

        # `out` has shape: [batch_size*num_slots, num_channels+1, height, width].
        assert_shape(out.size(), (batch_size * num_slots, nFrames, num_channels + 1, height, width))

        out = out.view(batch_size, num_slots, nFrames, num_channels + 1, height, width)
        recons = out[:, :, :, :num_channels, :, :]
        masks = out[:, :, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)
        recon_combined = torch.sum(recons * masks, dim=1)
        
        return recon_combined, recons, masks, slots_all

    def loss_function(self, input):
        recon_combined, recons, masks, slots_all = self.forward(input)
        loss_position = F.mse_loss(recon_combined, input)
        loss = loss_position
        return {
            "loss": loss,
            "loss_position_slot": loss_position,
        }


class SoftPositionEmbed(nn.Module):
    def __init__(self, num_channels: int, hidden_size: int, resolution: Tuple[int, int]):
        super().__init__()
        self.dense = nn.Linear(in_features=num_channels + 1, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))

    def forward(self, inputs: Tensor):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        assert_shape(inputs.shape[1:], emb_proj.shape[1:])
        return inputs + emb_proj