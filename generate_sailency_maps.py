import os.path
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lightning import Fabric

from tqdm import tqdm
from argparse import Namespace

from data import dataset

# from Models.DMRA.models import Model
from Models.HFIL.models import Model
# from Models.MAE.model import Model
# from Models.PLFRNet.models.PLFRNet import PLFRNet as Model

from utils import *


def get_all_data(path, model_name):
    img_path = os.path.join(path, "test_images")
    d_path = os.path.join(path, "test_depth")
    mask_path = os.path.join(path, "test_masks")
    seg_path = os.path.join(path, "test_segments")
    result_path = os.path.join(path, "result_{}".format(os.path.splitext(model_name)[0]))
    file_names = os.listdir(img_path)
    data_dict = {}
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for file_name in file_names:
        base_name = os.path.splitext(file_name)[0]
        data_dict.update({
            base_name: {
                "image": os.path.join(img_path, "{}.jpg".format(base_name)),
                "depth": os.path.join(d_path, "{}.png".format(base_name)),
                "mask": os.path.join(mask_path, "{}.png".format(base_name)),
                "segment": os.path.join(seg_path, "{}.jpg".format(base_name)),
                "result": os.path.join(result_path, "{}.png".format(base_name)),
            }
        })
    return data_dict


def get_batch(data_dict, device, dtype):
    img_size = [256, 256]
    tgt = cv.imread(data_dict["mask"]) / 255.0
    tgt = cv.resize(tgt, img_size)

    image = cv.imread(data_dict["image"]) / 255.0 * 2 - 1
    image = cv.resize(image, img_size)

    depth = cv.imread(data_dict["depth"]) / 255.0 * 2 - 1
    depth = depth if random.random() < 0.5 else -1 * depth
    depth = cv.resize(depth, img_size)

    seg_masks = torch.load(data_dict["segment"])
    seg_masks = np.array(seg_masks)
    h, w = seg_masks.shape[-2:]
    seg = np.ones([h, w, 3]) * np.random.random(3) * 255
    for mask in seg_masks[:-1]:
        mask = mask[..., None]
        seg = np.where(mask == 0, seg, mask * np.random.random(3) * 255)
    seg = cv.resize(seg, img_size) / 255.0 * 2 - 1

    train_data = {
        "image": torch.from_numpy(image[None, ...]).to(device=device, dtype=dtype),
        "depth": torch.from_numpy(depth[None, ...]).to(device=device, dtype=dtype),
        "tgt": torch.from_numpy(tgt[None, ...]).to(device=device, dtype=dtype),
        "seg": torch.from_numpy(seg[None, ...]).to(device=device, dtype=dtype)
    }
    return train_data


def val_one_data(model, batch, save_path):
    with torch.no_grad():
        loss, x, tgt, seg, img = model(batch)
        result = show_img(x[0].permute([1, 2, 0]), 'network')
        show_img(img[0].permute([1, 2, 0]), 'image')
        show_img(tgt[0].permute([1, 2, 0]), 'tgt')
        cv.imwrite(save_path, result)
    return get_F_beta(x, tgt), get_MAE(x, tgt)



def main(args):
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

    with fabric.init_module():
        model = Model(
            in_channels=[3, 3],
            out_channels=1,
            model_channels=512,
            num_blocks=8,
            patch_size=4
        )
        model.to(args.device)

    model = fabric.setup_module(model, move_to_device=True)

    state_dict = torch.load(args.model_name, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    data_dict = get_all_data(args.data_path, args.model_name)
    f_betas = []
    maes = []
    t = tqdm(data_dict.keys())
    for key in t:
        batch = get_batch(data_dict[key], args.device, model.dtype)
        save_path = data_dict[key]["result"]
        f_beta, mae = val_one_data(model, batch, save_path)
        f_betas.append(f_beta)
        maes.append(mae)
        t.set_postfix(fbeta=sum(f_betas)/len(f_betas))

if __name__ == "__main__":
    config = Namespace(
        project_name='NAS_SOD',
        wandb=False,
        train_mode="SD3",
        run_name='sd1',
        run_id=None,
        device='cuda:1',
        dtype='16-mixed',
        num_workers=1,
        epochs=2500,
        warmup_steps=1000,
        batch_size=1,
        step_size=1,
        optimizer='AdamW',
        lr=1e-5,
        data_path=
        # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\DUT-RGBD-Test",  # 0.9683 0.9478 0.0236
        # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\LFSD",  # 0.9173 0.8843 0.0587
        r"D:\Dataset\CV\NAS_SOD\RGBD_Test\NJUD",  # 0.9597 0.9305 0.0330
        # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\NLPR",  # 0.9694 0.9265 0.0200
        # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\SIP",  # 0.9289 0.9034 0.0460
        # r"D:\Dataset\CV\NAS_SOD\RGBD_Test\STERE",  # 0.9568 0.9082 0.0317
        model_name=r"best_DINO-NAS.ckpt",
    )

    main(config)
