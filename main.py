import argparse, os, logging, time
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch import optim

from datasets.ff import FFpp
from datasets.celeb_df import CelebDF
from datasets.dffd import DFFD
from datasets.dfdcp import DFDCP
from src.xception import xception_text

from xception import xception

from trainer import Trainer

from transform import TwoTransform, get_augs
from consistency_loss import ConsistencyCos, ConsistencyL2, ConsistencyL1

from utils import log_print

torch.multiprocessing.set_sharing_strategy("file_system")

def main(args):
    save_dir = os.path.join("ckpt",args.dataset, args.exp_name,args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    logfile = '{}/{}.log'.format(save_dir, time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    logging.basicConfig(filename=logfile, level=logging.INFO)
    logger = logging.getLogger()

    log_print("args: {}".format(args))

    # model
    if args.model_name == "xception":
        # model = xception(pretrained=True, num_classes=2)
        model = xception_text(pretrained=True, num_classes=2)

    else:
        raise NotImplementedError
    if torch.cuda.is_available():
        model = model.cuda()

    # transforms
    train_augs = get_augs(name=args.aug_name,norm=args.norm,size=args.size)
    if args.consistency != "None":
        train_augs = TwoTransform(train_augs)
    log_print("train aug:{}".format(train_augs))
    # test_augs = get_augs(name="None",norm=args.norm,size=args.size)
    # log_print("test aug:{}".format(test_augs))
    test_augs = get_augs(name="None",norm=args.norm,size=args.size)
    test_augs = TwoTransform(test_augs)
    log_print("test aug:{}".format(test_augs))

    # dataset
    if args.dataset == "ff":
        train_dataset = FFpp(args.root,"train",train_augs,num_classes=args.num_classes,quality=args.ff_quality)
        test_dataset = FFpp(args.root,"test",test_augs,num_classes=args.num_classes,quality=args.ff_quality)
    elif args.dataset == "celebdf":
        train_dataset = CelebDF(args.root,"train",train_augs)
        test_dataset = CelebDF(args.root,"test",test_augs)
    elif args.dataset == "dffd":
        train_dataset = DFFD(args.root,"train",train_augs)
        test_dataset = DFFD(args.root,"test",test_augs)
    elif args.dataset == "dfdcp":
        train_dataset = DFDCP(args.root,"train",train_augs)
        test_dataset = DFDCP(args.root,"test",test_augs)
    else:
        raise NotImplementedError

    log_print("len train dataset:{}".format(len(train_dataset)))
    log_print("len test dataset:{}".format(len(test_dataset)))
    # dataloader
    trainloader = DataLoader(train_dataset,
        batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers
    )
    testloader = DataLoader(test_dataset,
        batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = args.num_workers
    )

    if args.num_classes == 2:
        ce_weight = [args.real_weight, 1.0]
    else:
        raise NotImplementedError

    # CrossEntropy Loss
    weight=torch.Tensor(ce_weight)
    if torch.cuda.is_available():
        weight = weight.cuda()
    loss_fn = nn.CrossEntropyLoss(weight)

    if args.consistency == "None":
        consistency_fn = None
    else:
        # args.consistency = "cos"
        if args.consistency == "cos":
            consistency_fn = ConsistencyCos()
        elif args.consistency == "L2":
            consistency_fn = ConsistencyL2()
        elif args.consistency == "L1":
            consistency_fn = ConsistencyL1()
        else:
            raise NotImplementedError
        log_print("consistency loss function: {}, rate:{}".format(consistency_fn, args.consistency_rate))

    # optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(params=model.parameters(),lr=args.lr)
    else:
        raise NotImplementedError
    log_print("optimizer: {}".format(optimizer))

    if args.load_model_path is not None:
        log_print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.load_model_path) # , map_location="cpu"
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_recond = {
            "acc": checkpoint['acc'],
            "auc": checkpoint['auc'],
            "epoch": checkpoint['epoch'],
        }
        start_epoch = checkpoint['epoch'] + 1
        log_print("start from best recode: {}".format(best_recond))
    else:
        best_recond={"acc":0,"auc":0,"epoch":-1,"tdr3":0,"tdr4":0}
        start_epoch = 0

    # trainer
    trainer = Trainer(
        train_loader=trainloader, 
        test_loader=testloader,
        model=model, 
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        consistency_fn=consistency_fn,
        consistency_rate=args.consistency_rate,
        log_interval=args.log_interval, 
        best_recond=best_recond,
        save_dir=save_dir,
        exp_name=args.exp_name,
        amp=args.amp)

    for epoch_idx in range(start_epoch,args.epochs):
        print('-'*20 + 'Training' + '-'*20)
        trainer.train_epoch_text(epoch_idx)
        print('-'*20 + 'Testing' + '-'*20)
        trainer.test_epoch(epoch_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--num-classes', type=int, default=2)
    
    # consistency loss
    arg('--consistency', type=str, default="cos")
    arg('--consistency-rate', type=float, default=1.0)

    # transforms
    arg('--aug-name', type=str, default="RE")
    arg('--norm', type=str, default="0.5")
    arg('--size', type=int, default=299)

    # dataset 
    arg('--dataset', type=str, default='ff')
    arg('--ff-quality', type=str, default='c23',choices=['c23','c40','raw'])
    arg('--root', type=str, default='data/FF')
    arg('--batch-size', type=int, default=16)
    arg('--num-workers', type=int, default=8)
    arg('--shuffle', type=bool, default=True)

    arg('--real-weight', type=float, default=4.0)

    # optimizer
    arg('--optimizer', type=str, default="adam")
    arg('--lr', type=float, default=0.0002)

    arg('--exp-name', type=str, default='my_test_5')

    arg('--gpus', type=str, default='0')

    arg('--log-interval', type=int, default=500)

    arg("--epochs", type=int, default=40)
    # arg("--load-model-path", type=str, default='ckpt/ff/my_test_3/xception/epoch_24_acc_83.920_auc_77.316.pth')
    arg("--load-model-path", type=str, default=None)

    arg("--model-name", type=str, default="xception")

    arg("--amp", default=False, action='store_true')

    arg("--seed", type=int, default=3407) # https://arxiv.org/abs/2109.08203

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed) 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
