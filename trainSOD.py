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
# from Models.HFIL.HFILNet import HFILNet as Model

from utils import *



def train_epoch(fabric, model, network_optimizer, dataloader, epoch, epochs, total_step, last_step, writer, args):
    t = tqdm(dataloader)
    f_measures_network = []
    mae_network = []
    for batch in t:
        loss, x, tgt, seg, img = model(batch)
        fabric.backward(loss)
        network_optimizer.step()
        network_optimizer.zero_grad()

        f_measures_network.append(get_F_beta(x, tgt))
        mae_network.append(get_MAE(x, tgt))
        # mae_network.append(loss.detach().cpu().numpy())
        show_img(x[0].permute([1, 2, 0]), 'network')
        show_img(tgt[0].permute([1, 2, 0]), 'tgt_network')
        # show_img(seg[0].permute([1, 2, 0]), 'seg_network')
        # show_img(img[0].permute([1, 2, 0]), 'img_network')

        mean_F = sum(f_measures_network) / len(f_measures_network)
        mean_MAE = sum(mae_network) / len(mae_network)

        t.set_postfix(
            epoch='{}/{}'.format(epoch, epochs),
            F_network=mean_F,
            MAE=mean_MAE
        )
        total_step += 1
        writer.add_scalar('Loss_DM/train', loss.item(), total_step + last_step)

    writer.add_scalar('MAE/train', mean_MAE, epoch)
    writer.add_scalar('F_beta/train', mean_F.item(), epoch)
    return mean_MAE, total_step

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
            img2 = show_img(tgt[0].permute([1, 2, 0]), 'tgt_network')
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

    # cv.imwrite("NLPR.jpg", np.concatenate(imgs[:10], axis=0))
    # print(np.concatenate(imgs, axis=0).shape)
    # print("image saved")
    writer.add_scalar('MAE/val', mean_MAE.item(), epoch)
    writer.add_scalar('F_beta/val', mean_F.item(), epoch)
    return mean_F, mean_MAE

def train(args):
    import torch.backends.cuda
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    torch.set_float32_matmul_precision('high')
    # torch.manual_seed(3407)
    fabric = Fabric(
        accelerator='cuda',
        precision=args.dtype,
        devices=[0 if args.device == "cuda:0" else 1]
    )
    fabric.launch()

    writer = SummaryWriter()

    data_generator = dataset(path=args.data_path, mode="train")
    dataloader = DataLoader(
        data_generator,
        batch_size=args.batch_size,
        shuffle=True,
        persistent_workers=args.num_workers != 0,
        num_workers=args.num_workers,
        pin_memory=True)

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

    optimizer = torch.optim.AdamW(lr=args.lr, params=[
        {'params': model.parameters(), 'initial_lr': args.lr},
    ])

    model = fabric.setup_module(model, move_to_device=True)
    optimizer = fabric.setup_optimizers(optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)
    test_dataloader = fabric.setup_dataloaders(test_dataloader)

    best_f_measure = 0
    last_step = 0
    last_epoch = 0
    if os.path.exists("best_" + args.model_name):
        state_dict = torch.load("best_" + args.model_name, map_location="cpu")
        model.load_state_dict(state_dict["state_dict"], strict=True)
        best_f_measure = state_dict["best_f_measure"]
        print(best_f_measure)


    model.train()
    total_step = 0
    for epoch in range(0, epochs):
        f_measures_network, total_step = train_epoch(
            fabric,
            model, optimizer, dataloader,
            epoch, epochs, total_step, last_step,
            writer,
            args
        )

        test_F, test_MAE= val_epoch(
            model, test_dataloader, epoch, writer
        )

        if test_F > best_f_measure:
            best_f_measure = test_F
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_f_measure": best_f_measure,
                    "last_step": last_step + total_step,
                    "epoch": last_epoch + epoch
                },
                "best_" + args.model_name
            )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "best_f_measure": best_f_measure,
                "last_step": last_step + total_step,
                "epoch": last_epoch + epoch
            },
            args.model_name
        )


if __name__ == "__main__":
    config = Namespace(
        project_name='NAS_SOD',
        wandb=False,
        train_mode="SD3",
        run_name='sd1',
        run_id=None,
        device='cuda:0',
        dtype='16-mixed',
        num_workers=1,
        epochs=2500,
        warmup_steps=1000,
        batch_size=1,
        step_size=1,
        optimizer='AdamW',
        lr=1e-5,
        data_path=[
            r"D:\Dataset\CV\NAS_SOD\RGBD_Train",
            # r"D:\Dataset\CV\NAS_SOD\DUTS-TR",
        ],
        test_path=[
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\DUT-RGBD-Test",  # 0.9771 0.9628 0.0178
            # r"D:\Dataset\CV\NAS_SOD\RGBD_/Test\LFSD",  # 0.9332 0.9019 0.0454
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\NJUD",  # 0.9658 0.9407 0.0178
            r"D:\Dataset\CV\NAS_SOD\RGBD_Test\NLPR",  # 0.9727 0.9287 0.0171
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\SIP",  # 0.9482 0.9219 0.0364
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\STERE",  # 0.9603 0.9163 0.0273
        ],
        model_name="new-DINO-NAS.ckpt",
        save_every_n_epoch=1,
    )

    train(config)
