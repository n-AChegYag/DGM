import torch
import torch.nn as nn
from models import discriminator, encoder_decoder
from losses import my_loss
import numpy as np
import cv2
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim
import utils
import matplotlib.pyplot as plt
from torchvision.utils import save_image


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    np.random.seed(426)
    torch.manual_seed(426)
    torch.cuda.manual_seed_all(426)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    DATA_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/eye'
    LOG_PATH = '/home/ps/disk12t/ACY/AD/MIA/log/220515_2'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_dict = {'loss_1':[], 'loss_2':[]}

    batch_size = 16
    lr = 1e-5
    num_workers = 4
    epochs = 500
    lambda_A = 10
    lambda_R = 10
    lambda_TV = 1
    train_dis_freq = 5
    print_freq = 20
    save_freq = 20
    best_loss_1 = np.inf
    best_loss_2 = np.inf

    transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor()
    ])

    # dataset
    eye_dataset = ImageFolder(root=DATA_PATH, transform=transform)

    # dataloader
    eye_dataloader = DataLoader(eye_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # losses
    TV_Loss = my_loss.TVLoss()
    # R_Loss = nn.MSELoss()
    R_Loss = nn.L1Loss()
    GAN_Loss = my_loss.GANLoss()

    # models
    E_G = encoder_decoder.Encoder(norm='in')
    E_F = encoder_decoder.Encoder(norm='none')
    D_G = encoder_decoder.Decoder()
    D_F = encoder_decoder.Decoder()
    D_J = encoder_decoder.Decoder_j()
    D = discriminator.MultiscaleDiscriminator(1)

    E_G.to(device)
    E_F.to(device)
    D_G.to(device)
    D_F.to(device)
    D_J.to(device)
    D.to(device)

    
    part_1 = [E_G, E_F, D_G, D_F, D_J]
    part_2 = [D]

    optim_params_1 = [{'params': net.parameters()} for net in part_1]
    optimizer_1 = optim.Adam(optim_params_1, lr)

    optim_params_2 = [{'params': net.parameters()} for net in part_2]
    optimizer_2 = optim.Adam(optim_params_1, lr)

    loss_1_am = utils.AverageMeter()
    loss_2_am = utils.AverageMeter()

    plt.figure(figsize=(10, 6))
    for epoch in range(1, epochs+1):
        loss_1_am.reset()
        loss_2_am.reset()
        for idx, (image, label) in enumerate(eye_dataloader):
            image = (image - 0.5) * 2
            image = image[:,0,:,:].unsqueeze(1)
            # image, label = image.to(device), label.to(device)
            data_dict, num_N = utils.choose_N_A_data(image, label, batch_size)
            image_N = data_dict['N']
            image_A = data_dict['A']
            if image_N != None:
                image_N = image_N.to(device)
                c_z = E_G(image_N).detach()
                y_p = D_G(c_z).detach()
                loss_g = GAN_Loss(y_p, True)
                fake_N = D(y_p)
                loss_d_fake = GAN_Loss(fake_N, False)
                real_N = D(image_N)
                loss_d_real = GAN_Loss(real_N, True)
                loss_2 = loss_d_fake + loss_d_real
                loss_2_am.update(loss_2.item(), num_N)
                optimizer_2.zero_grad()   
                loss_2.backward()
                optimizer_2.step()
                if idx % print_freq == 0:
                    print(
                        'D: [{}] | [{} / {}]'.format(epoch, idx, len(eye_dataloader)),
                        'loss: {:.4f} ({:.4f})'.format(loss_2_am.val, loss_2_am.avg),
                        'lr: {:.6f}'.format(optimizer_2.param_groups[0]['lr']),
                        sep='\n'
                    )
            if idx % train_dis_freq == 0:
                image = image.to(device)
                c_z = E_G(image)
                y_p = D_G(c_z)
                loss_tv = TV_Loss(y_p)
                c_s = E_F(image)
                a = D_F(c_s)
                a_label = torch.zeros_like(a).to(device)
                for i, ii in enumerate(label):
                    if ii.item() == 0:
                        a_label[i,:,:,:] = a[i,:,:,:]
                z_pp = D_J(c_z, c_s)
                z_p = y_p + a
                loss_r1 = R_Loss(z_p, image)
                loss_r2 = R_Loss(z_pp, image)
                loss_r3 = R_Loss(a, a_label)
                loss_1 = lambda_A*loss_g + lambda_R*(loss_r1 + loss_r2 + loss_r3) + lambda_TV*loss_tv
                loss_1_am.update(loss_1.item(), batch_size)
                optimizer_1.zero_grad()   
                loss_1.backward()
                optimizer_1.step()
                if idx % print_freq == 0:
                    print(
                        'G: [{}] | [{} / {}]'.format(epoch, idx, len(eye_dataloader)),
                        'loss: {:.4f} ({:.4f})'.format(loss_1_am.val, loss_1_am.avg),
                        'lr: {:.6f}'.format(optimizer_1.param_groups[0]['lr']),
                        sep='\n'
                    )
        if epoch % save_freq == 0:
            save_image(y_p / 2 + 0.5, os.path.join(LOG_PATH, 'y_p_{}.jpg'.format(epoch)))
            save_image(z_pp / 2 + 0.5, os.path.join(LOG_PATH, 'z_pp_{}.jpg'.format(epoch)))
            save_image(z_p / 2 + 0.5, os.path.join(LOG_PATH, 'z_p_{}.jpg'.format(epoch)))
            save_image(a, os.path.join(LOG_PATH, 'a_{}.jpg'.format(epoch)))

        # save checkpoint
        loss_dict['loss_1'].append(loss_1_am.avg)
        loss_dict['loss_2'].append(loss_2_am.avg)
        utils.plot_loss_curve(loss_dict['loss_1'], loss_dict['loss_2'], LOG_PATH)
        np.save(os.path.join(LOG_PATH, 'loss.npy'), loss_dict) 

        if loss_1_am.avg < best_loss_1:
            best_loss_1 = loss_1_am.avg
            is_best = True
        else:
            is_best = False
        state = {
            'E_G': E_G.state_dict(),
            'E_F': E_F.state_dict(),
            'D_G': D_G.state_dict(),
            'D_F': D_F.state_dict(),
            'D_J': D_J.state_dict(),
            'D':   D.state_dict(),
        }
        utils.save_checkpoint(state=state, is_best=is_best, save_path=LOG_PATH)









    
