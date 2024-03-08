#-*- coding:utf-8 -*-
import math
from mindspore import Tensor
import mindspore
import mindspore.nn as nn
from mindspore import ops

def to_2tuple(num):
    return (num, num)

def _trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    rand_tensor = ops.uniform(tensor.shape, Tensor(2 * l - 1), Tensor(2 * u - 1))
    tensor = rand_tensor

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv()

    # Transform to proper mean, std
    tensor = tensor * (std * math.sqrt(2.))
    tensor = tensor + mean

    # Clamp to ensure it's in the proper range
    tensor = ops.clip_by_value(tensor, min=Tensor(a), max=Tensor(b))
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _trunc_normal_(tensor, mean, std, a, b)

def drop_path1(x, drop_prob=0., training=False, scale_by_keep=True):
    if not Tensor(drop_prob) == Tensor(0.) or not training:
        return x
    keep_prob = 1 - drop_prob
    random_tensor = ops.zeros_like(x).bernoulli(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor



class DropPath(nn.Cell):
    def __init__(self, drop_prob=0, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def construct(self, x):
        return drop_path1(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(p=drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.transpose(0, 1, 3, 2, 4, 5).view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.transpose(0, 1, 3, 2, 4, 5).view(B, H, W, -1)
    return x


class WindowAttention(nn.Cell):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = mindspore.Parameter(
            ops.Zeros()(((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads), mindspore.float32))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = Tensor(mindspore.numpy.arange(self.window_size[0]))
        coords_w = Tensor(mindspore.numpy.arange(self.window_size[1]))
        coords = ops.Stack()(ops.Meshgrid(indexing="xy")((coords_h, coords_w)))  # 2, Wh, Ww
        coords_flatten = ops.Flatten()(coords)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.relative_position_index = relative_position_index

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)

        self.proj_drop = nn.Dropout(p=proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).view(B_, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = ops.BatchMatMul()(q, k.swapaxes(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(2, 0, 1)  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.expand_dims(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.expand_dims(1).expand_dims(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = ops.BatchMatMul()(attn, v).swapaxes(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class LocalContextExtractor(nn.Cell):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.conv = nn.SequentialCell(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, padding=0, has_bias=True),
            nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, pad_mode='pad', padding=1, has_bias=True),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, padding=0, has_bias=True),
            nn.LeakyReLU(alpha=0.2),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.SequentialCell(
            nn.Dense(dim, dim // reduction, has_bias=False),
            nn.ReLU(),
            nn.Dense(dim // reduction, dim, has_bias=False),
            nn.Sigmoid()
        )

    def construct(self, x):
        x = self.conv(x)
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


class ContextAwareTransformer(nn.Cell):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer([dim])
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.attn_mask = attn_mask

        self.lce = LocalContextExtractor(self.dim)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = ops.Zeros()((1, H, W, 1), mindspore.float32)  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.expand_dims(1) - mask_windows.expand_dims(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def construct(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        L = L
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # local context features
        lcf = x.transpose(0, 3, 1, 2)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = ops.roll(x, (-self.shift_size, -self.shift_size), (1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = ops.roll(shifted_x, (self.shift_size, self.shift_size), (1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        # local context
        lc = self.lce(lcf)
        lc = lc.view(B, C, H * W).transpose(0, 2, 1)
        x = lc + x

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Cell):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.CellList([
            ContextAwareTransformer(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def construct(self, x, x_size):
        for blk in self.blocks:
            x = blk(x, x_size) # B L C
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class ContextAwareTransformerBlock(nn.Cell):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.dilated_conv = nn.Conv2d(dim, dim, kernel_size=3, pad_mode='pad', padding=2, has_bias=True, dilation=2)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.dilated_conv = nn.SequentialCell(
                nn.Conv2d(dim, dim // 4, kernel_size=3, pad_mode='pad', padding=2, dilation=2),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, padding=0),
                nn.LeakyReLU(alpha=0.2),
                nn.Conv2d(dim // 4, dim, kernel_size=3, pad_mode='pad', padding=2, dilation=2)
                )

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def construct(self, x, x_size):
        res = self.residual_group(x, x_size) # B L C
        res = self.patch_unembed(res, x_size) # B c H W
        res = self.dilated_conv(res)
        res = self.patch_embed(res) + x
        return res

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class PatchEmbed(nn.Cell):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer((embed_dim,))
        else:
            self.norm = None

    def construct(self, x):
        B, C, H, W = x.shape
        H = H
        W = W
        x = x.view(B, C, -1).swapaxes(1, 2)  # B C H W ==> B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Cell):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def construct(self, x, x_size):
        B, HW, C = x.shape
        HW = HW
        C = C
        x = x.swapaxes(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class SpatialAttentionModule(nn.Cell):

    def __init__(self, dim):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.relu = nn.LeakyReLU()

    def construct(self, x1, x2):
        f_cat = ops.Concat(axis=1)((x1, x2))
        att_map = ops.Sigmoid()(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class HDRTransformer(nn.Cell):

    def __init__(self, img_size=128, patch_size=1, in_chans=6,
                 embed_dim=60, depths=None, num_heads=None,
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, resi_connection='1conv'):
        super(HDRTransformer, self).__init__()
        num_in_ch = in_chans
        num_out_ch = 3
        ################################### 1. Feature Extraction Network ###################################
        # coarse feature
        self.conv_f1 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, pad_mode='pad', padding=1)
        self.conv_f2 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, pad_mode='pad', padding=1)
        self.conv_f3 = nn.Conv2d(num_in_ch, embed_dim, 3, 1, pad_mode='pad', padding=1)
        # spatial attention module
        self.att_module_l = SpatialAttentionModule(embed_dim)
        self.att_module_h = SpatialAttentionModule(embed_dim)
        self.conv_first = nn.Conv2d(embed_dim * 3, embed_dim, 3, 1, pad_mode='pad', padding=1)
        ################################### 2. HDR Reconstruction Network ###################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = mindspore.Parameter(ops.Zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.asnumpy() for x in ops.LinSpace()(Tensor(0.), Tensor(drop_path_rate), sum(depths))]  # stochastic depth decay rule

        self.layers = nn.CellList([ContextAwareTransformerBlock(dim=embed_dim,
                                                                input_resolution=(patches_resolution[0],
                                                                                  patches_resolution[1]),
                                                                depth=depths[i_layer],
                                                                num_heads=num_heads[i_layer],
                                                                window_size=window_size,
                                                                mlp_ratio=self.mlp_ratio,
                                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                                drop_path=dpr[sum(depths[:i_layer]): \
                                                                sum(depths[:i_layer + 1])],
                                                                norm_layer=norm_layer,
                                                                downsample=None,
                                                                use_checkpoint=use_checkpoint,
                                                                img_size=img_size,
                                                                patch_size=patch_size,
                                                                resi_connection=resi_connection
                                                                ) for i_layer in range(self.num_layers)])
        self.norm = norm_layer([self.num_features])

        # build the last conv layer
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, pad_mode='pad', padding=1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.SequentialCell(nn.Conv2d(embed_dim, \
                                                        embed_dim // 4, 3, 1, pad_mode='pad', padding=1),
                                                     nn.LeakyReLU(alpha=0.2),
                                                     nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, padding=0),
                                                     nn.LeakyReLU(alpha=0.2),
                                                     nn.Conv2d(embed_dim // 4, embed_dim, \
                                                     3, 1, pad_mode='pad', padding=1))

        self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, pad_mode='pad', padding=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # x [B, embed_dim, h, w]
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # B L C
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)
        return x

    def construct(self, x1, x2, x3):
        # feature extraction network
        # coarse feature
        f1 = self.conv_f1(x1)
        f2 = self.conv_f2(x2)
        f3 = self.conv_f3(x3)

        # spatial feature attention
        f1_att_m = self.att_module_h(f1, f2)
        f1_att = f1 * f1_att_m
        f3_att_m = self.att_module_l(f3, f2)
        f3_att = f3 * f3_att_m
        x = self.conv_first(ops.Concat(axis=1)((f1_att, f2, f3_att)))

        # CTBs for HDR reconstruction
        res = self.conv_after_body(self.forward_features(x) + x)
        x = self.conv_last(f2 + res)
        x = ops.Sigmoid()(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            i = i
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        return flops
