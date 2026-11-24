import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from functools import partial
from torch.optim import lr_scheduler
from .sfm import SaliencyResampler, AdaptiveUpsampler, FAN


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == "syncbatch":
        norm_layer = functools.partial(nn.SyncBatchNorm, affine=True, track_running_stats=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == "cosine":
        t_max = int(getattr(opt, "cosine_tmax", 0) or 0) or opt.n_epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    # Modules named in net._keep_init_prefixes hold pretrained weights and must
    # not be re-initialized (used by the ResNet-encoder U-Net).
    keep = tuple(getattr(net, "_keep_init_prefixes", ()))
    for name, module in net.named_modules():
        if name and any(name == p or name.startswith(p + ".") for p in keep):
            continue
        init_func(module)


def init_net(net, init_type="normal", init_gain=0.02):
    """Initialize a network: 1. register CPU/GPU device; 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """
    import os

    if torch.cuda.is_available():
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            net.to(local_rank)
            print(f"Initialized with device cuda:{local_rank}")
        else:
            net.to(0)
            print("Initialized with device cuda:0")
    init_weights(net, init_type, init_gain=init_gain)
    # ARS start-up mode, set by define_G: 0 = force the identity map (this port's
    # previous behaviour), 1 = keep the random init so the grid can move.
    if int(getattr(net, "_sfm_sal_init", 0)) == 0 and hasattr(net, "reset_sfm_to_identity"):
        net.reset_sfm_to_identity()
    return net


def define_G(input_nc, output_nc, ngf, netG, norm="batch", use_dropout=False, init_type="normal", init_gain=0.02, antialias=False, sfm_levels=0, sfm_base_size=512, sfm_start_level=1, sfm_msau=0, sfm_fan=0, sfm_fan_out=0, sfm_pad=0, sfm_fwhm=0, sfm_sal_init=0):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_128 | unet_256
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        antialias (bool)   -- U-Net only: low-pass filter before each stride (BlurPool)

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == "resnet_9blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == "resnet_6blocks":
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == "unet_128":
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, antialias=antialias, sfm_levels=sfm_levels, sfm_base_size=sfm_base_size, sfm_start_level=sfm_start_level, msau=bool(sfm_msau), fan=bool(sfm_fan), fan_out=bool(sfm_fan_out))
    elif netG == "unet_256":
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, antialias=antialias, sfm_levels=sfm_levels, sfm_base_size=sfm_base_size, sfm_start_level=sfm_start_level, msau=bool(sfm_msau), fan=bool(sfm_fan), fan_out=bool(sfm_fan_out))
    elif netG == "resnet_unet":
        net = ResNetUNetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                    sfm_levels=sfm_levels, sfm_base_size=sfm_base_size,
                                    sfm_start_level=sfm_start_level, msau=bool(sfm_msau), fan=bool(sfm_fan))
    else:
        raise NotImplementedError("Generator model name [%s] is not recognized" % netG)
    # --- ARS configuration hooks (see models/sfm.py) -------------------------
    # sfm_pad / sfm_fwhm: override the Gaussian-centroid search window. The released
    #   SFM FPN uses padding_size=5, fwhm=3; this port sizes it as level/32.
    # sfm_sal_init: stored on the net and consumed by init_net, because init_weights
    #   only runs there. 0 = identity start (previous behaviour), 1 = random init.
    if sfm_pad or sfm_fwhm:
        for m in net.modules():
            if isinstance(m, SaliencyResampler):
                m.set_window(sfm_pad or m.padding_size, sfm_fwhm or m.fwhm)
    net._sfm_sal_init = int(sfm_sal_init)
    return net


def define_D(input_nc, ndf, netD, n_layers_D=3, norm="batch", init_type="normal", init_gain=0.02):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == "basic":  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == "n_layers":  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == "pixel":  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError("Discriminator model name [%s] is not recognized" % netD)
    return net


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == "real":  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError(f"{type} not implemented")
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv, grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias), norm_layer(ngf * mult * 2), nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias), norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ConvBlock(nn.Module):
    """(conv -> norm -> relu) x 2."""

    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, 3, padding=1, bias=False),
            norm_layer(output_nc),
            nn.ReLU(True),
            nn.Conv2d(output_nc, output_nc, 3, padding=1, bias=False),
            norm_layer(output_nc),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """2x upsample, concatenate the encoder skip, then (conv -> norm -> relu) x 2.

    The upsampled tensor is cropped to the skip size so that inputs whose side is
    not a multiple of 32 (inference tiles such as 340) still line up.

    ``unwarp`` is applied to the upsampled tensor before the skip is concatenated.
    With ARS active the encoder side of a skip lives in a warped coordinate frame,
    so the decoder output has to be brought back into that same frame first;
    otherwise the concat mixes two geometries, which is exactly the blur MSAU is
    meant to avoid.
    """

    def __init__(self, input_nc, skip_nc, output_nc, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, unwarp=None, fan=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_nc, output_nc, 2, 2)
        self.conv = ConvBlock(output_nc + skip_nc, output_nc, norm_layer)
        self.drop = nn.Dropout(0.5) if use_dropout else None
        self.unwarp = unwarp
        self.fan = FAN(output_nc, norm_layer) if fan else None

    def forward(self, x, skip):
        y = self.up(x)
        y = y[:, :, : skip.shape[2], : skip.shape[3]]
        if self.unwarp is not None:
            y = self.unwarp(y)
        y = self.conv(torch.cat([y, skip], 1))
        if self.fan is not None:
            y = self.fan(y)
        return self.drop(y) if self.drop is not None else y


class ResNetUNetGenerator(nn.Module):
    """U-Net whose encoder is a ResNet34, optionally ImageNet-pretrained.

    The LightMyCells input is a single gray plane, so it is replicated to three
    channels before entering the RGB encoder. The pretrained encoder is listed in
    ``_keep_init_prefixes`` so ``init_net`` does not overwrite the loaded weights.
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, pretrained=True, sfm_levels=0, sfm_base_size=512,
                 sfm_start_level=1, msau=False, fan=False):
        super().__init__()
        import torchvision.models as tvm

        try:
            weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = tvm.resnet34(weights=weights)
        except Exception as exc:  # no offline node: fall back to random init
            print("resnet34 pretrained weights unavailable (%s); training from scratch" % exc)
            resnet = tvm.resnet34(weights=None)

        self.input_nc = input_nc
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # /2, 64
        self.pool = resnet.maxpool                                        # /4
        self.layer1 = resnet.layer1                                       # /4, 64
        self.layer2 = resnet.layer2                                       # /8, 128
        self.layer3 = resnet.layer3                                       # /16, 256
        self.layer4 = resnet.layer4                                       # /32, 512
        self._keep_init_prefixes = ("stem", "pool", "layer1", "layer2", "layer3", "layer4")

        # ImageNet statistics: the pretrained encoder's BN running stats describe
        # mean/std-normalised RGB inputs. Feeding raw [0,1] (or [-1,1]) tensors
        # shifts every activation, which showed up as a constant +0.23 offset in
        # the output and PSNR 14 instead of ~20.
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        scale = max(ngf, 1) / 64.0
        w = [max(int(round(c * scale)), 8) for c in (256, 128, 64, 64)]

        def sfm_for(level):
            if level >= sfm_levels or level < sfm_start_level:
                return None
            size = sfm_base_size / (2.0 ** level)
            return int(size) if size >= 32 else None

        def make_ars(channels, level):
            size = sfm_for(level)
            if size is None:
                return None
            pad = max(2, int(round(float(size) / 32.0)))
            return SaliencyResampler(channels, padding_size=pad, fwhm=pad,
                                     psp_ratio=16, align_corners=False)

        # ARS sits after a ResNet stage, so its channel count must match that
        # stage's output: layer1=64, layer2=128, layer3=256, layer4=512.
        # The level index is the log2 of the resolution divisor, because
        # sfm_for() sizes the ARS kernel from base_size / 2**level.
        self.ars_8 = make_ars(128, 3)    # after layer2, /8  (e.g. 64px at 512 in)
        self.ars_16 = make_ars(256, 4)   # after layer3, /16 (e.g. 32px)
        self.msau = AdaptiveUpsampler(align_corners=False) if msau else None
        self.fan_enabled = fan

        # up4 leaves layer4's /32 frame and concatenates s8, which ARS warped at
        # /16, so it must be un-warped with that grid. Likewise up3 with ars_8.
        # up2 concatenates the never-warped /4 skip, so it needs no un-warp.
        self.up4 = UpBlock(512, 256, w[0], norm_layer, use_dropout,
                           unwarp=partial(self._unwrap, ars=self.ars_16), fan=fan)  # /32 -> /16
        self.up3 = UpBlock(w[0], 128, w[1], norm_layer, use_dropout,
                           unwarp=partial(self._unwrap, ars=self.ars_8), fan=fan)   # /16 -> /8
        self.up2 = UpBlock(w[1], 64, w[2], norm_layer, use_dropout, fan=fan)        # /8  -> /4
        self.up1 = UpBlock(w[2], 64, w[3], norm_layer, use_dropout)                 # /4  -> /2
        self.up0 = nn.Sequential(             # /2  -> /1
            nn.ConvTranspose2d(w[3], w[3], 2, 2),
            nn.ReLU(True),
            nn.Conv2d(w[3], output_nc, 3, padding=1),
            nn.Tanh(),
        )

    def reset_sfm_to_identity(self):
        for m in self.modules():
            if isinstance(m, SaliencyResampler):
                m.reset_to_identity()

    def _warp(self, x, ars):
        return x if ars is None else ars(x)

    def _unwrap(self, x, ars):
        """Undo one ARS warp (MSAU). Skipped when ARS or MSAU is disabled."""
        if ars is None or self.msau is None or not hasattr(ars, "last_grid"):
            return x
        gh, gw = ars.last_grid.shape[2:]
        if x.shape[2] != gh or x.shape[3] != gw:
            return x
        return self.msau(x, ars.last_grid)

    def forward(self, x):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
        x = (x - self.img_mean) / self.img_std
        s1 = self.stem(x)                    # /2,  64
        s2 = self.layer1(self.pool(s1))      # /4,  64
        s4 = self._warp(self.layer2(s2), self.ars_8)    # /8,  128
        s8 = self._warp(self.layer3(s4), self.ars_16)   # /16, 256
        y = self.layer4(s8)                             # /32, 512
        y = self.up4(y, s8)
        y = self.up3(y, s4)
        y = self.up2(y, s2)
        y = self.up1(y, s1)
        return self.up0(y)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, antialias=False, sfm_levels=0, sfm_base_size=512, sfm_start_level=1, msau=False, fan=False, fan_out=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            antialias (bool) -- low-pass before each stride (BlurPool)
            sfm_levels (int) -- equip levels [sfm_start_level, sfm_levels) with
                                Adaptive Resampling
            sfm_base_size (int) -- input side length, used to size the ARS kernels
            sfm_start_level (int) -- first level to equip; the default of 1 skips
                                the raw input image, where ARS is both very
                                expensive and pointless (1 channel, no features)

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()

        def sfm_for(level):
            """Side length of this level, or None when ARS is not wanted."""
            if level >= sfm_levels or level < sfm_start_level:
                return None
            size = sfm_base_size / (2.0 ** level)
            return int(size) if size >= 32 else None

        # construct unet structure; the innermost block is the deepest level
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, antialias=antialias, msau=msau, fan=fan)  # add the innermost layer
        depth = num_downs - 1
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, antialias=antialias, sfm_size=sfm_for(depth - i), msau=msau, fan=fan)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, antialias=antialias, sfm_size=sfm_for(3), msau=msau, fan=fan)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, antialias=antialias, sfm_size=sfm_for(2), msau=msau, fan=fan)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, antialias=antialias, sfm_size=sfm_for(1), msau=msau, fan=fan)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, antialias=antialias, sfm_size=sfm_for(0), msau=msau, fan=fan, fan_out=fan_out)  # add the outermost layer

    def reset_sfm_to_identity(self):
        """Start every ARS warp as the identity map (see SaliencyResampler)."""
        for m in self.modules():
            if isinstance(m, SaliencyResampler):
                m.reset_to_identity()

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class BlurPool2d(nn.Module):
    """Anti-aliased subsampling (Zhang, "Making Convolutional Networks Shift-Invariant Again").

    A strided convolution reads every k-th pixel, so anything above the new
    Nyquist frequency folds back into the feature map as aliasing. Low-pass
    filtering *before* the stride removes that band instead of letting it wrap
    around. The filter is a fixed binomial kernel (no learnable parameters), so
    it only changes how detail is discarded, not model capacity.
    """

    def __init__(self, filt_size=3, stride=2, channels=None):
        super(BlurPool2d, self).__init__()
        self.stride = stride
        self.filt_size = filt_size
        if filt_size == 3:
            k = torch.tensor([1.0, 2.0, 1.0])
        elif filt_size == 4:
            k = torch.tensor([1.0, 3.0, 3.0, 1.0])
        elif filt_size == 5:
            k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            raise NotImplementedError("BlurPool2d supports filt_size in (3, 4, 5)")
        k = k / k.sum()
        kernel = (k[:, None] * k[None, :])[None, None]
        self.channels = channels
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        c = x.shape[1]
        kernel = self.kernel.expand(c, 1, self.filt_size, self.filt_size)
        pad = self.filt_size // 2
        return F.conv2d(x, kernel, stride=self.stride, padding=(pad, pad), groups=c)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, antialias=False, sfm_size=None, msau=False, fan=False, fan_out=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            antialias (bool)    -- low-pass filter before the stride (BlurPool)
            sfm_size (int|None) -- input side length at this level; when set, an
                                   Adaptive Resampling (ARS) block warps the
                                   features before the stride so salient detail
                                   survives downsampling
            msau (bool)         -- un-warp the decoder output with the ARS grid
                                   (Multi-Scale Adaptive Upsampling), so the
                                   reconstruction happens in uniform coordinates
            fan (bool)          -- frequency adaptive normalization on the merged
                                   decoder feature
            fan_out (bool)      -- additionally apply FAN to the full-resolution
                                   prediction itself, i.e. a learnable
                                   frequency-dependent gain on the output
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        if antialias:
            # stride-1 conv keeps the resolution, the blur performs the subsampling
            downconv = nn.Sequential(
                nn.Conv2d(input_nc, inner_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
                BlurPool2d(filt_size=3, stride=2, channels=inner_nc))
        else:
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        resampler = None
        if sfm_size:
            # keep the centroid search radius at a constant fraction of the map
            pad = max(2, int(round(float(sfm_size) / 32.0)))
            resampler = SaliencyResampler(input_nc, padding_size=pad, fwhm=pad,
                                          psp_ratio=16, align_corners=False)
        pre = [] if resampler is None else [resampler]
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = pre + [downconv]
            # Tanh moved into forward() so fan_out can sit between upconv and the
            # squash; the parameter names of this block are unchanged.
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu] + pre + [downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu] + pre + [downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        # Remember where the ARS module sits so forward() can reuse its grid on the
        # skip tensor. Storing the index instead of the module avoids registering
        # the same parameters twice in state_dict.
        self._sfm_pos = model.index(resampler) if resampler is not None else -1
        # MSAU only makes sense together with ARS: it consumes that level's grid.
        self.msau = AdaptiveUpsampler(align_corners=False) if (resampler is not None and msau) else None
        # innermost is still concatenated (skip x with the decoded output), so the
        # merged channel count is 2*outer_nc there as well. At the outermost level
        # there is no concat, so FAN sees the single-channel prediction.
        if outermost:
            self.fan = FAN(outer_nc, norm_layer) if fan_out else None
        else:
            self.fan = FAN(outer_nc * 2, norm_layer) if fan else None

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            out = self.model(x)
            if self.fan is not None:
                out = self.fan(out)
            return torch.tanh(out)
        out = self.model(x)
        if self._sfm_pos >= 0:
            res = self.model[self._sfm_pos]
            if self.msau is not None:
                # MSAU: undo the warp so the decoder returns to uniform coordinates,
                # matching the (never warped) skip tensor below.
                gh, gw = res.last_grid.shape[2:]
                if out.shape[2] == gh and out.shape[3] == gw:
                    out = self.msau(out, res.last_grid)
                x = self._unwarp(x, res)
            else:
                # No MSAU: warp the skip so both sides of the concat share the
                # warped frame produced by ARS.
                x = res.warp_with(x, res.last_grid)
        out = torch.cat([x, out], 1)
        if self.fan is not None:
            out = self.fan(out)
        return out

    def _unwarp(self, x, res):
        """Bring the skip tensor into the frame MSAU left the decoder in.

        MSAU maps the warped decoder output back to uniform coordinates, and the
        skip is already uniform, so nothing to do. Kept as a hook so the two
        branches stay symmetric if the frame convention changes.
        """
        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
