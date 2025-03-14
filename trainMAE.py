from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lightning import Fabric

import os
from tqdm import tqdm
from argparse import Namespace

from data import dataset

# from Models.DMRA.models import Model
# from Models.HFIL.models import Model
from Models.MAE.model import Model

from utils import *



def train_epoch(fabric, model, network_optimizer, dataloader, epoch, epochs, total_step, last_step, writer, args):
    t = tqdm(dataloader)
    f_measures_network = []
    mae_network = []
    for batch in t:
        loss, img, x, depth, y = model(batch)
        fabric.backward(loss)
        network_optimizer.step()
        network_optimizer.zero_grad()

        f_measures_network.append(get_F_beta(x, img))
        mae_network.append(get_MAE(x, img))
        # mae_network.append(loss.detach().cpu().numpy())
        show_img(img[0].permute([1, 2, 0]), 'img')
        show_heatmap(img[0].permute([1, 2, 0]), x[0].permute([1, 2, 0]), 'x')
        show_img(depth[0].permute([1, 2, 0]), 'depth')
        show_heatmap(depth[0].permute([1, 2, 0]), y[0].permute([1, 2, 0]), 'y')

        mean_F = sum(f_measures_network) / len(f_measures_network)
        if len(mae_network) > 100:
            mae_network = mae_network[1:]
        mean_MAE = sum(mae_network) / len(mae_network)

        t.set_postfix(
            epoch='{}/{}'.format(epoch, epochs),
            F_network=loss.item(),
            MAE=loss.item()
        )
        total_step += 1
        writer.add_scalar('Loss_DM/train', loss.item(), total_step + last_step)

        if (total_step + 1) % 1000 == 0:
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "full_model": model.state_dict()
                },
                args.model_name
            )
            print("model_saved")

    writer.add_scalar('MAE/train', loss.item(), epoch)
    writer.add_scalar('F_beta/train', loss.item(), epoch)
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
    return mean_MAE, mean_MAE

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
        # model = Model(
        #     in_channels=[3, 3],
        #     out_channels=1,
        #     model_channels=512,
        #     num_blocks=4,
        #     patch_size=4
        # )
        # model = HFILNet()
        # model = EncoderDecoder(is_test=True)
        model = Model(
            model_channels=768,
            patch_size=16,
            num_blocks=4,
            mask_rate=0,
            predict_dim=65536
        )
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
    if os.path.exists(args.model_name):
        state_dict = torch.load(args.model_name, map_location="cpu")
    #     model.s_encoder.load_state_dict(state_dict["s_encoder"], strict=True)
    #     model.t_encoder.load_state_dict(state_dict["t_encoder"], strict=True)
    #     model.decoder.load_state_dict(state_dict["decoder"], strict=True)
        model.load_state_dict(state_dict["full_model"], strict=False)

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
        # test_F, test_MAE= val_epoch(
        #     model, test_dataloader, epoch, writer
        # )

        torch.save(
            {
                # "t_encoder": model.t_encoder.state_dict(),
                "encoder": model.encoder.state_dict(),
                # "decoder": model.decoder.state_dict(),
                "full_model": model.state_dict()
            },
            args.model_name
        )
        # if test_F > best_f_measure:
        #     best_f_measure = test_F
        #     torch.save(
        #         {
        #             "state_dict": model.state_dict(),
        #             "best_f_measure": best_f_measure,
        #             "last_step": last_step + total_step,
        #             "epoch": last_epoch + epoch
        #         },
        #         "best_" + args.model_name
        #     )


if __name__ == "__main__":
    config = Namespace(
        project_name='NAS_SOD',
        wandb=False,
        train_mode="SD3",
        run_name='sd1',
        run_id=None,
        device='cuda:0',
        dtype='16-mixed',
        num_workers=8,
        epochs=2500,
        warmup_step=1000,
        batch_size=128,
        step_size=1,
        optimizer='AdamW',
        lr=5e-5,
        data_path=[
            # r"D:\Dataset\CV\ImageNet\ILSVRC2012_img_train\train_depths",
            r"D:\Dataset\CV\ImageNet\ILSVRC2012_img_train\train_images",
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Train",
        ],
        test_path=[
            r"D:\Dataset\CV\NAS_SOD\RGBD_Test\DUT-RGBD-Test",  # 0.9562 0.9206 0.0374
            # r"D:\Dataset\CV\NAS_SOD\dataset\RGBD-SOD\RGBD_Test\LFSD",  # 0.8665 0.8057 0.1105
            # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\NJUD",  # 0.9418 0.8549 0.0540
            # r"D:\Dataset\CV\NAS_SOD\dataset\RGBD-SOD\RGBD_Test\NLPR",  # 0.9774 0.9096 0.0186
            # r"D:\Dataset\CV\NAS_SOD\dataset\RGBD-SOD\RGBD_Test\SIP",  # 0.9543 0.90119 0.0395
            # r"D:\Dataset\CV\NAS_SOD\dataset\RGBD-SOD\RGBD_Test\STERE",  # 0.9664 0.9089 0.0296
        ],
        model_name="CrossMAE.ckpt",
        save_every_n_epoch=1,
    )

    train(config)
