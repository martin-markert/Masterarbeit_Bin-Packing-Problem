import torch
import torch.nn as nn

class Spatial_Positional_Encoding(nn.Module):
    def __init__(self, dim_model, bin_size_x, bin_size_y):
        super().__init__()

        if dim_model < 2 or dim_model % 2 != 0:
                raise ValueError(f"dim_model must be >= 2 and divisible by 2, got {dim_model}")
        
        dummy_tensor = torch.ones((1, bin_size_x, bin_size_y))

        x_embed = dummy_tensor.cumsum(2, dtype=torch.float32)
        y_embed = dummy_tensor.cumsum(1, dtype=torch.float32)

        pos_features = dim_model // 2
        dimension_tensor = torch.arange(pos_features, dtype=torch.float32)
        dimension_tensor = 10000 ** (2 * (torch.div(dimension_tensor, 2, rounding_mode='trunc') / pos_features))

        pos_x = x_embed[:, :, :, None] / dimension_tensor
        pos_y = y_embed[:, :, :, None] / dimension_tensor

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).flatten(1, 2)
        self.register_buffer('pos', pos)

    def forward(self, x):
        return x + self.pos

# Test
dim_model = 1
bin_size_x = 2
bin_size_y = 2

batch_size = 2

seq_len = bin_size_x * bin_size_y


model = Spatial_Positional_Encoding(dim_model, bin_size_x, bin_size_y)

dummy_input = torch.rand(batch_size, seq_len, dim_model)


output = model(dummy_input)

print("Output shape:", output.shape)
print("Output:\n", output)
