import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks


def _gaussian_window(size, sigma, device, dtype):
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def ssim_loss(pred, target, size=11, sigma=1.5, data_range=2.0):
    """1 - SSIM for tensors in [-1, 1] (data_range=2), averaged over the batch."""
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    win = _gaussian_window(size, sigma, pred.device, pred.dtype)
    n, c = pred.shape[:2]
    x = pred.reshape(n * c, 1, *pred.shape[2:])
    y = target.reshape(n * c, 1, *target.shape[2:])
    pad = size // 2

    def filt(t):
        return F.conv2d(t, win, padding=pad)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = filt(x * x) - mu_x2
    sig_y = filt(y * y) - mu_y2
    sig_xy = filt(x * y) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / \
               ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return 1.0 - ssim_map.mean()


def pcc_loss(pred, target, eps=1e-8):
    """1 - Pearson correlation per sample (shape invariance to brightness/contrast)."""
    n = pred.shape[0]
    x = pred.reshape(n, -1)
    y = target.reshape(n, -1)
    xc = x - x.mean(1, keepdim=True)
    yc = y - y.mean(1, keepdim=True)
    corr = (xc * yc).sum(1) / (xc.norm(dim=1) * yc.norm(dim=1) + eps)
    return (1.0 - corr).mean()


def _diff_mag(x):
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1])[:, :, :-1, :]
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :])[:, :, :, :-1]
    return dx, dy


def gradient_loss(pred, target, mode="deficit", margin=0.0, eps=1e-6):
    """High-frequency term.

    mode="diff":      L1/gradient-matching. Note this is minimised by the
                      *median* gradient, which for sparse thin structures is
                      ~0, so it quietly rewards even smoother predictions.
    mode="deficit":   only penalise missing edge energy, relu(|grad_t| - |grad_p|),
                      i.e. ask the prediction to be at least as sharp as the GT
                      without capping extra detail. This is the form that can
                      actually undo the conditional-mean blur.
    """
    dpx, dpy = _diff_mag(pred)
    dtx, dty = _diff_mag(target)
    if mode == "diff":
        return (dpx - dtx).abs().mean() + (dpy - dty).abs().mean()
    mp = torch.sqrt(dpx * dpx + dpy * dpy + eps)
    mt = torch.sqrt(dtx * dtx + dty * dty + eps)
    return torch.relu(mt - margin - mp).mean()


def cosine_distance(pred, target, eps=1e-8):
    """1 - cosine similarity on the raw (not mean-centred) image vectors.

    Unlike PCC this keeps the DC term, so it also penalises a prediction whose
    overall brightness is off - the dominant failure mode we observed.
    """
    n = pred.shape[0]
    x = pred.reshape(n, -1)
    y = target.reshape(n, -1)
    cos = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + eps)
    return (1.0 - cos).mean()


class Pix2PixModel(BaseModel):
    """This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm="batch", netG="unet_256", dataset_mode="aligned")
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode="vanilla")
            parser.add_argument("--lambda_L1", type=float, default=100.0,
                                help="weight of the reconstruction term (L1 or MSE, see --pix_loss)")
            parser.add_argument("--pix_loss", default="l1", choices=("l1", "mse", "smoothl1"),
                                help="reconstruction loss; the LightMyCells winners use MSE, the "
                                     "ISBI'24 Bag-of-Experts entry uses SmoothL1")
            parser.add_argument("--smoothl1_beta", type=float, default=0.2,
                                help="--pix_loss smoothl1: residual magnitude (in the network's "
                                     "-1..1 units) above which the loss becomes linear instead of "
                                     "quadratic. PyTorch's default of 1.0 never leaves the quadratic "
                                     "regime for [-1,1] targets, which would make SmoothL1 identical "
                                     "to 0.5*MSE; 0.2 matches the 75th percentile of the residual "
                                     "on the mitochondria task (~24/255 gray levels)")
            parser.add_argument("--ssim_weight", type=float, default=0.0,
                                help="weight of (1 - SSIM); 0 disables it")
            parser.add_argument("--pcc_weight", type=float, default=0.0,
                                help="weight of (1 - Pearson correlation); 0 disables it")
            parser.add_argument("--cd_weight", type=float, default=0.0,
                                help="weight of the cosine distance; 0 disables it")
            parser.add_argument("--grad_weight", type=float, default=0.0,
                                help="weight of the high-frequency (gradient) term; "
                                     "counteracts the blur inherent to pixel/SSIM losses")
            parser.add_argument("--grad_mode", default="deficit", choices=("deficit", "diff"),
                                help="deficit: only penalise missing edge energy (default); "
                                     "diff: plain gradient L1 (its optimum is the median "
                                     "gradient, i.e. flat output)")
            parser.add_argument("--grad_margin", type=float, default=0.0,
                                help="deficit mode: tolerated shortfall before the term kicks in")
            parser.add_argument("--antialias", type=int, default=0,
                                help="1: low-pass (BlurPool) before each U-Net stride, which stops "
                                     "detail from aliasing into the feature maps instead of being "
                                     "cleanly discarded")
            parser.add_argument("--grad_clip", type=float, default=0.0,
                                help="max global grad norm for G; 0 disables clipping. Useful for "
                                     "the ARS/SFM path, whose sampling grid can produce spikes")
            parser.add_argument("--sfm_msau", type=int, default=0,
                                help="1: add MSAU, the inverse of the ARS warp, so the decoder "
                                     "reconstructs in uniform coordinates instead of the warped one")
            parser.add_argument("--sfm_fan", type=int, default=0,
                                help="1: add Frequency Adaptive Normalization, which normalises the "
                                     "low- and high-frequency branches separately")
            parser.add_argument("--sfm_fan_out", type=int, default=0,
                                help="1: also apply FAN to the full-resolution prediction, i.e. a "
                                     "learnable frequency-dependent gain on the output")
            parser.add_argument("--sfm_levels", type=int, default=0,
                                help="N>0: insert Adaptive Resampling (SFM/TPAMI 2025) before the "
                                     "stride of the N finest U-Net levels; it warps salient detail "
                                     "into a lower band so it survives downsampling")
            parser.add_argument("--sfm_base_size", type=int, default=512,
                                help="input side length, used to size the ARS Gaussian kernels")
            parser.add_argument("--sfm_pad", type=int, default=0,
                                help="override the ARS centroid search radius in pixels "
                                     "(0 = auto: level size / 32). The released SFM FPN uses 5")
            parser.add_argument("--sfm_fwhm", type=int, default=0,
                                help="override the ARS Gaussian width in pixels (0 = same as pad). "
                                     "The released SFM FPN uses 3")
            parser.add_argument("--sfm_sal_init", type=int, default=0,
                                help="1 = keep the saliency head at its random init (official "
                                     "behaviour, the sampling grid can move); 0 = force the "
                                     "identity map at start-up (previous default here)")
            parser.add_argument("--sfm_start_level", type=int, default=1,
                                help="first U-Net level to equip with ARS; 1 skips the raw "
                                     "input image (only 1 channel, and by far the slowest level)")
            parser.add_argument("--lambda_GAN", type=float, default=1.0,
                                help="weight of the adversarial loss; 0 disables the "
                                     "discriminator entirely (no D is built or updated)")

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.use_gan = float(getattr(opt, "lambda_GAN", 1.0)) > 0
        self.pix_loss = getattr(opt, "pix_loss", "l1")
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        if self.use_gan:
            self.loss_names += ["G_GAN"]
        self.loss_names += ["G_rec"]
        if getattr(opt, "ssim_weight", 0) > 0:
            self.loss_names += ["G_SSIM"]
        if getattr(opt, "pcc_weight", 0) > 0:
            self.loss_names += ["G_PCC"]
        if getattr(opt, "cd_weight", 0) > 0:
            self.loss_names += ["G_CD"]
        if getattr(opt, "grad_weight", 0) > 0:
            self.loss_names += ["G_Grad"]
        if self.use_gan:
            self.loss_names += ["D_real", "D_fake"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ["real_A", "fake_B", "real_B"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ["G", "D"] if self.use_gan else ["G"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        self.device = opt.device
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, antialias=bool(int(getattr(opt, "antialias", 0) or 0)), sfm_levels=int(getattr(opt, "sfm_levels", 0) or 0), sfm_base_size=int(getattr(opt, "sfm_base_size", 512) or 512), sfm_start_level=int(getattr(opt, "sfm_start_level", 1) or 0), sfm_msau=int(getattr(opt, "sfm_msau", 0) or 0), sfm_fan=int(getattr(opt, "sfm_fan", 0) or 0), sfm_fan_out=int(getattr(opt, "sfm_fan_out", 0) or 0), sfm_pad=int(getattr(opt, "sfm_pad", 0) or 0), sfm_fwhm=int(getattr(opt, "sfm_fwhm", 0) or 0), sfm_sal_init=int(getattr(opt, "sfm_sal_init", 0)))

        if self.isTrain and self.use_gan:  # define a discriminator; conditional GANs need to take both input and output images
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            # define loss functions
            if self.use_gan:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            if self.pix_loss == "mse":
                self.criterionRec = torch.nn.MSELoss()
            elif self.pix_loss == "smoothl1":
                self.criterionRec = torch.nn.SmoothL1Loss(
                    beta=float(getattr(opt, "smoothl1_beta", 0.2) or 0.2))
            else:
                self.criterionRec = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.use_gan:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Generator objective: reconstruction (+ optional structure terms, optional GAN)."""
        self.loss_G = 0
        if self.use_gan:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * float(self.opt.lambda_GAN)
            self.loss_G = self.loss_G + self.loss_G_GAN
        self.loss_G_rec = self.criterionRec(self.fake_B, self.real_B) * self.opt.lambda_L1
        self.loss_G = self.loss_G + self.loss_G_rec
        if getattr(self.opt, "ssim_weight", 0) > 0:
            self.loss_G_SSIM = ssim_loss(self.fake_B, self.real_B) * self.opt.ssim_weight
            self.loss_G = self.loss_G + self.loss_G_SSIM
        if getattr(self.opt, "pcc_weight", 0) > 0:
            self.loss_G_PCC = pcc_loss(self.fake_B, self.real_B) * self.opt.pcc_weight
            self.loss_G = self.loss_G + self.loss_G_PCC
        if getattr(self.opt, "cd_weight", 0) > 0:
            self.loss_G_CD = cosine_distance(self.fake_B, self.real_B) * self.opt.cd_weight
            self.loss_G = self.loss_G + self.loss_G_CD
        if getattr(self.opt, "grad_weight", 0) > 0:
            self.loss_G_Grad = gradient_loss(
                self.fake_B, self.real_B,
                mode=getattr(self.opt, "grad_mode", "deficit"),
                margin=float(getattr(self.opt, "grad_margin", 0.0))) * self.opt.grad_weight
            self.loss_G = self.loss_G + self.loss_G_Grad
        self.loss_G.backward()
        clip = float(getattr(self.opt, "grad_clip", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), clip)

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        if self.use_gan:
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # update G's weights
