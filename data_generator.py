import h5py,os
import numpy as np
from torch.utils.data import DataLoader,Dataset
import torch
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import LinfPGDAttack,GradientSignAttack,L2PGDAttack,SpatialTransformAttack,JacobianSaliencyMapAttack,MomentumIterativeAttack
import torch.nn  as nn
from torchvision import models , transforms , datasets
from attackers import  *


class ImageNetDataset(Dataset):
    """Dataset wrapping tensors.
    """

    def __init__(self, data, target):
        super(ImageNetDataset, self).__init__()
        self.data = data
        self.target = target

    def __getitem__(self, index): # 该函数涉及到enumerate的返回值
        batch_x = self.data[index]
        batch_y = self.target[index]
        return batch_x, batch_y

    def __len__(self):
        return self.data.size(0)

def _generate_random_small_test_set(set_size=1000):
    # load data
    data_dir = "/data/liyanjie/Tiny-ImageNet/Tiny-ImageNet/tiny-imagenet-200/"
    data_transforms = transforms.Compose([transforms.ToTensor()])
    print("Initializing Datasets and DataLoaders...")
    test_datasets = datasets.ImageFolder(os.path.join(data_dir, "val"), data_transforms)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=50, shuffle=True, num_workers=4)

    torch.manual_seed(0)
    data = np.zeros([set_size,3,64,64])
    true_target = np.zeros([set_size])

    # perturb
    for batch_idx,(clndata, target) in enumerate(test_loader):
        if batch_idx * 50 >= set_size:
            print('data loaded with :{} samples'.format(batch_idx * 50))
            break
        data[batch_idx*50:(batch_idx+1)*50,:,:,:] = clndata.numpy()
        true_target[batch_idx*50:(batch_idx+1)*50 ] = target.numpy()

    print("test_true_target.shape:{}".format(true_target.shape))

    h5_store = h5py.File("data/test_tiny_ImageNet_" + str(set_size) + ".h5", 'w')
    h5_store.create_dataset('data', data=data)
    h5_store.create_dataset('true_target', data=true_target)
    h5_store.close()



def get_test_adv_loader(attack_method,epsilon,batch_size,shuffle):
    #save file
    if os.path.exists("data/test_tiny_ImageNet_adv_"+str(attack_method)+"_"+str(epsilon)+".h5"):
        h5_store = h5py.File("data/test_tiny_ImageNet_adv_"+str(attack_method)+"_"+str(epsilon)+".h5", 'r')
        test_data = h5_store['data'][:] # 通过切片得到np数组
        test_true_target=h5_store['true_target'][:]
        h5_store.close()
    else:
        pass

    # 生成dataset的包装类
    train_data = torch.from_numpy(test_data)
    train_target = torch.from_numpy(test_true_target)  # numpy转Tensor
    train_dataset = ImageNetDataset(train_data, train_target)
    del train_data,train_target
    return DataLoader(dataset=train_dataset, num_workers=4, drop_last=True, batch_size=batch_size,
                  shuffle=shuffle)



def _generate_adv_file(attack_method,num_classes,epsilon,set_size):
    # load model
    model = torch.load(os.path.join("checkpoint", "resnet50_epoch_22.pth"))
    model = model.cuda()

    #define attack
    if attack_method == "PGD":
        adversary = LinfPGDAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=20, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0,
            targeted=False)
    elif attack_method == "FGSM":
        adversary = GradientSignAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            clip_min=0.0, clip_max=1.0, eps=epsilon, targeted=False)
    elif attack_method == "Momentum":
        adversary = MomentumIterativeAttack(
            model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=20, decay_factor=1.0, eps_iter=1.0, clip_min=0.0, clip_max=1.0,
            targeted=False, ord=np.inf)
    elif attack_method == "STA":
        adversary = SpatialTransformAttack(
            model, num_classes=num_classes, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            initial_const=0.05, max_iterations=500, search_steps=1, confidence=0, clip_min=0.0, clip_max=1.0,
            targeted=False, abort_early=True)
    elif attack_method == "DeepFool":
        adversary = DeepFool(model, max_iter=20, clip_max=1.0, clip_min=0.0, epsilon=epsilon)
    elif attack_method == "CW":
        adversary = CarliniWagnerL2Attack(
            model, num_classes=args.num_classes, epsilon=epsilon, loss_fn=nn.CrossEntropyLoss(reduction="sum"),
            max_iterations=20, confidence=0, clip_min=0.0, clip_max=1.0,
            targeted=False, abort_early=True)



    # version two
    h5_store = h5py.File("data/test_tiny_ImageNet_" + str(set_size) + ".h5","r")
    data = h5_store['data'][:]
    target = h5_store['true_target'][:]
    data = torch.from_numpy(data)
    target = torch.from_numpy(target)
    test_dataset = ImageNetDataset(data, target)
    test_loader =  DataLoader(dataset=test_dataset, num_workers=4, drop_last=True, batch_size=50,
                  shuffle=False)


    torch.manual_seed(0)
    test_adv = np.zeros([set_size,3,64,64])
    test_true_target = np.zeros([set_size])

    # perturb
    for batch_idx,(clndata, target) in enumerate(test_loader):
        print("{}/{}".format(batch_idx,set_size//50))
        clndata, target = clndata.cuda().float(), target.cuda().long()
        with ctx_noparamgrad_and_eval(model):
            # print(target)
            advdata = adversary.perturb(clndata, target)
            test_adv[batch_idx*50:(batch_idx+1)*50,:,:,:] = advdata.detach().cpu().numpy()
        test_true_target[batch_idx*50:(batch_idx+1)*50]=target.cpu().numpy()

    print("test_adv.shape:{}".format(test_adv.shape))
    print("test_true_target.shape:{}".format(test_true_target.shape))
    del model

    h5_store = h5py.File("data/test_tiny_ImageNet_"+str(set_size)+"_adv_" + str(attack_method) + "_" + str(epsilon) + ".h5", 'w')
    h5_store.create_dataset('data', data=test_adv)
    h5_store.create_dataset('true_target', data=test_true_target)
    h5_store.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Imagenet')
    parser.add_argument('--seed', default=0, type=int)
    # attack
    parser.add_argument("--attack_method", default="PGD", type=str,
                        choices=['FGSM', 'PGD','Momentum','STA','DeepFool','CW'])

    parser.add_argument('--epsilon', type = float,default=8/255, help='if pd_block is used')
    parser.add_argument('--num_classes', default=1000, type=int)

    # test
    parser.add_argument('--test_samples', default=1000, type=int)
    args = parser.parse_args()
    _generate_adv_file(attack_method=args.attack_method, num_classes=200, epsilon=args.epsilon,set_size=args.test_samples)
    # for attack_method in ['PGD','FGSM','Momentum']:
    #     for epsilon in [0.00784,0.03137,0.06275]:
    #         _generate_adv_file(attack_method=attack_method,num_classes=1000,epsilon=epsilon)
    # _generate_adv_file(attack_method='STA', num_classes=1000, epsilon='0.0')
    # _generate_random_small_test_set()
    # _generate_random_small_test_set(1000)