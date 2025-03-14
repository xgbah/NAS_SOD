import os.path

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lightning import Fabric

from tqdm import tqdm
from argparse import Namespace

from data import dataset

# from Models.DMRA.models import Model
from Models.HFIL.models import Model
# from Models.MAE.model import Model
# from Models.EATNet.models.model import Detector as Model
# from Models.PLFRNet.models.PLFRNet import PLFRNet as Model

from utils import *


def val_epoch(model, dataloader, epoch, writer):
    with torch.no_grad():
        t = tqdm(dataloader)
        f_measures_network = []
        mae_network = []
        E = []
        S = []
        imgs = []
        for batch in t:
            loss, x, tgt, seg, img = model(batch)

            f_measures_network.append(get_F_beta(x, tgt))
            mae_network.append(get_MAE(x, tgt))
            S.append(get_S_measure(x, tgt))
            E.append(get_E_measure(x, tgt))
            img1 = show_img(x[0].permute([1, 2, 0]), 'network')
            img2 = show_img(tgt[0].permute([1, 2, 0]), 'tgt_network', waitKey=1)
            img = np.concatenate([img1, img2], axis=1)
            imgs.append(img)

            mean_F = sum(f_measures_network) / len(f_measures_network)
            mean_MAE = sum(mae_network) / len(mae_network)
            mean_S = sum(S) / len(S)
            mean_E = sum(E) / len(E)

            t.set_postfix(
                F_network=mean_F,
                MAE=mean_MAE,
                S=mean_S,
                E=mean_E
            )

    writer.add_scalar('MAE/val', mean_MAE.item(), epoch)
    writer.add_scalar('F_beta/val', mean_F.item(), epoch)
    return mean_F, mean_MAE

def train(args):
    import torch.backends.cuda
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_float32_matmul_precision('high')
    fabric = Fabric(
        accelerator='cuda',
        precision=args.dtype,
        devices=[0 if args.device == "cuda:0" else 1]
    )
    fabric.launch()

    writer = SummaryWriter()

    test_generator = dataset(path=args.test_path, mode="test")
    test_dataloader = DataLoader(
        test_generator,
        batch_size=args.batch_size,
        shuffle=True,
        persistent_workers=args.num_workers != 0,
        num_workers=args.num_workers,
        pin_memory=True
    )
    epochs = args.epochs

    with fabric.init_module():
        model = Model(
            in_channels=[3, 3],
            out_channels=1,
            model_channels=512,
            num_blocks=8,
            patch_size=4
        )
        # model = Model()
        model.to(args.device)

    model = fabric.setup_module(model, move_to_device=True)
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    state_dict = torch.load(args.model_name, map_location="cpu")
    model.load_state_dict(state_dict["state_dict"], strict=True)

    for epoch in range(0, epochs):
        test_F, test_MAE= val_epoch(
            model, test_dataloader, epoch, writer
        )


if __name__ == "__main__":
    config = Namespace(
        project_name='NAS_SOD',
        wandb=False,
        train_mode="SD3",
        run_name='sd1',
        bun_id=None,
        device='cuda:0',
        dtype='16-mixed',
        num_workers=1,
        epochs=10,
        warmup_steps=1000,
        batch_size=1,
        step_size=1,
        optimizer='AdamW',
        lr=1e-5,
        data_path=[
            # r"D:\Dataset\CV\ImageNet\ILSVRC2012_img_train\train_depths",
            # r"D:\Dataset\CV\ImageNet\ILSVRC2012_img_train\train_images",
            r"D:\Dataset\CV\NAS_SOD\RGBD_Train",
            # r"D:\Dataset\CV\NAS_SOD\DUTS-TR",
        ],
        test_path=[
            r"D:\Dataset\CV\NAS_SOD\RGBD_Test\DUT-RGBD-Test",  # 0.9683 0.9478 0.0236
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\LFSD",  # 0.9173 0.8843 0.0587
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\NJUD",  # 0.9597 0.9305 0.0330
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\NLPR",  # 0.9694 0.9265 0.0200
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\SIP",  # 0.9289 0.9034 0.0460
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\STERE",  # 0.9568 0.9082 0.0317
        ],
        model_name=r".\best_new-DINO-NAS.ckpt",
        save_every_n_epoch=1,
    )

    train(config)
