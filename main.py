import argparse
import transformers_robust
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import time
import os
import sys
time_str = time.strftime('%Y-%m-%d-%H-%M')

path = ''

parser = argparse.ArgumentParser(description='Train classification network')

parser.add_argument('--model',type=str, default='vit')
parser.add_argument('--norm',type=str, default='L2')
parser.add_argument('--gamma',type=float, default=4.0)
parser.add_argument('--delta',type=float, default=9.0)
parser.add_argument('--epsilon',type=float, default=1e-2)
parser.add_argument('--L',type=int, default=3)
parser.add_argument('--batch_size',type=int, default=8)
parser.add_argument('--attack',type=str, default='fgsm')#Self tf ‘adv_eval_0’
parser.add_argument('--budget',type=int, default=1)

args = parser.parse_args()
args.budget = args.budget/255



def load_params(args, model):
    
    i = 0
    if args.model == 'vit':
        layers = model.vit.encoder.layer
    elif args.model == 'beit':
        layers = model.beit.encoder.layer
    elif args.model == 'swin':
        layers = model.swin.encoder.layers
        
   
    if args.model == 'swin':
        for layer in layers:
            for block in layer.blocks:
                block.attention.self.robust_sum.L = args.L
                block.attention.self.robust_sum.norm = args.norm
                block.attention.self.robust_sum.gamma = args.gamma
                block.attention.self.robust_sum.delta = args.delta
                block.attention.self.robust_sum.epsilon = args.epsilon
                i +=1
        print(f"Only replace {i} layers with robust attention")
    else:
        for layer in layers:
            
            layer.attention.attention.robust_sum.L = args.L
            layer.attention.attention.robust_sum.norm = args.norm
            layer.attention.attention.robust_sum.gamma = args.gamma
            layer.attention.attention.robust_sum.delta = args.delta
            layer.attention.attention.robust_sum.epsilon = args.epsilon
            i +=1

            
        print(f"Only replace {i} layers with robust attention")
    
    return model

class LinfPGDAttack(object):
    def __init__(self, model, epsilon = 0.0314, k = 7, alpha = 0.00784):
    #def __init__(self, model, epsilon = 0.0314, k = 20, alpha = 0.003):
        
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha =alpha

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x).logits
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x
    
class FGSMAttack(object):
    def __init__(self, model, epsilon=0.0314):
        self.model = model
        self.epsilon = epsilon

    def perturb(self, x_natural, y):
        # 复制数据，避免影响原始数据
        x = x_natural.detach().clone()

        # 计算梯度
        x.requires_grad_()
        with torch.enable_grad():
            logits = self.model(x).logits
            loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0]

        # 应用FGSM公式：x' = x + epsilon * sign(grad)
        x_adv = x + self.epsilon * torch.sign(grad.detach())

        # 裁剪以保持在合法的数据范围内（比如图像数据通常在0到1之间）
        x_adv = torch.clamp(x_adv, 0, 1)

        return x_adv

def test(test_loader, net, criterion):
    if (args.attack in ['pgd','adv']) or 'adv_eval' in args.attack:
        adversary = LinfPGDAttack(net,  epsilon = args.budget, k = 7, alpha = 0.00784)
    elif args.attack == 'fgsm':
        adversary = FGSMAttack(net,  epsilon = args.budget)
        

    print('\n[ Test Start ]')
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):

            
        inputs, targets = inputs.cuda(), targets.cuda()
        total += targets.size(0)
        #print(targets)

        outputs = net(inputs).logits
        loss = criterion(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()
        #print(predicted)
        if batch_idx % 10 == 0:
            print(f'\nCurrent batch:{str(batch_idx)}' )
            print(f'Current benign test accuracy:{str(predicted.eq(targets).sum().item() / targets.size(0))}')
            print(f'Current benign test loss:{loss.item()}')

        adv = adversary.perturb(inputs, targets)

        adv_outputs = net(adv).logits
        loss = criterion(adv_outputs, targets)
        adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()
        #print(predicted)
        
        if batch_idx % 10 == 0:
            print(f'Current adversarial test accuracy:{str(predicted.eq(targets).sum().item() / targets.size(0))}' )
            print(f'Current adversarial test loss:{ loss.item()}')
            print(f'Total benign test accuarcy:{100. * benign_correct / total}')
            print(f'Total adversarial test Accuarcy:{100. * adv_correct / total}')
        
            
    #plt.imshow(np.transpose(inputs[1].cpu().numpy()[1]))
    #plt.savefig(path+"1.png")

    #os.path.join(final_output_dir, "robust_pgd", "train_set")
    #torch.save(torch.cat(advs),path+f"cifar10_sdnet18/robust_pgd/train_set/advs_{adversary.epsilon}.pth")
    print(f'Total benign test accuarcy:{100. * benign_correct / total}')
    print(f'Total adversarial test Accuarcy:{100. * adv_correct / total}')
    print(f'Total benign test loss:{benign_loss}')
    print(f'Total adversarial test loss:{adv_loss}')

    return 100. * benign_correct / total, 100. * adv_correct / total



def adv_train(epoch, train_loader, model,  optimizer, criterion):
    
    
    adversary = LinfPGDAttack(model)
    print('\n[ Train epoch: %d ]' % epoch)
    model.train()
    
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        adv = adversary.perturb(inputs, targets)
        adv_outputs = model(adv).logits
        loss = criterion(adv_outputs, targets)
        loss.backward()

        optimizer.step()
        train_loss += loss.item()
        _, predicted = adv_outputs.max(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Current batch:{str(batch_idx)}')
            print(f'Current adversarial train accuracy:{str(predicted.eq(targets).sum().item() / targets.size(0))}')
            print(f'Current adversarial train loss:{loss.item()}')
            print(f'Total adversarial train accuarcy:{ 100. * correct / total}')
            print(f'Total adversarial train loss:{train_loss}')

    print(f'Total adversarial train accuarcy:{ 100. * correct / total}')
    print(f'Total adversarial train loss:{train_loss}')
    return 100. * correct / total
    

def main():
    print(args)


    # 加载预训练的ViT-Small模型
    if args.model == 'vit':
        model = transformers_robust.ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10').cuda()
    elif args.model == 'deit':
        model = transformers_robust.DeiTForImageClassification.from_pretrained('tzhao3/DeiT-CIFAR10').cuda()
    elif args.model == 'beit':
        model = transformers_robust.BeitForImageClassification.from_pretrained('SajjadAlam/beit_Cifar10_finetune_model').cuda()
    elif args.model == 'swin':
        model = transformers_robust.SwinForImageClassification.from_pretrained('Weili/swin-base-patch4-window7-224-in22k-finetuned-cifar10').cuda()
        # model = transformers_robust.SwinForImageClassification.from_pretrained('Weili/swin-tiny-patch4-window7-224-finetuned-cifar10').cuda()

        

    #model = nn.DataParallel(model)



    model = load_params(args, model)
    
    transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),  #
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #normalize,
            transforms.Resize(224),
            transforms.CenterCrop(224),
        ])

    # 准备数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        #lambda x: feature_extractor(x, return_tensors='pt')['pixel_values'].squeeze(0),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        
    ])

    # 这里使用CIFAR10作为示例，你可以替换成其他数据集
    if args.attack == 'adv':
        train_dataset = datasets.CIFAR10(root=path+'data', train=True, download=True, transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = datasets.CIFAR10(root=path+'data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    if args.attack == 'adv':
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
        for epoch in range(0, 50):
            adv_train(epoch, train_loader, model, optimizer, criterion)
            torch.save(model.state_dict(),  path+f'log_cv/{args.model}/{args.attack}/'+f'state_epoch_{epoch}.pth.tar')
    
    if 'adv_eval' in args.attack:
        epoch = args.attack.split("_")[2]
        model.eval()
        model.load_state_dict(torch.load(path+f'log_cv/{args.model}/adv/'+f'state_epoch_{epoch}.pth.tar'))
        
        
    # 设置模型为评估模式  
    model.eval()

    criterion = nn.CrossEntropyLoss().cuda()
    test(test_loader, model, criterion=criterion)


    print("-----end------")



if __name__ == '__main__':
    os.makedirs(path+f'log_cv/{args.model}/{args.attack}', exist_ok=True)
    sys.stdout = open(path+f'log_cv/{args.model}/{args.attack}/norm{args.norm}_gamma{args.gamma}_delta{args.delta}_epsilon{args.epsilon}_L{args.L}_budget{args.budget}_{time_str}.log', 'w', buffering=1)
    main()
    sys.stdout.close()
    
