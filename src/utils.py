import os
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np

def check_and_create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def plot_loss_curve(loss_1, loss_2, save_path):
    '''
    precondition: plt.figure(figsize=(10, 6))
    '''
    y1 = np.array(loss_1)
    y2 = np.array(loss_2)
    x1 = range(0,len(y1))
    x2 = range(0,len(y2))
    plt.plot(x1, y1, '', label='loss_1')
    plt.plot(x2, y2, '', label='loss_2')
    plt.title('loss')
    plt.legend(loc='upper right')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.clf()

def save_checkpoint(state, is_best, save_path):
    torch.save(state['E_G'], os.path.join(save_path,'E_G.pkl'))
    torch.save(state['E_F'],os.path.join(save_path,'E_F.pkl'))
    torch.save(state['D_G'], os.path.join(save_path,'D_G.pkl'))
    torch.save(state['D_F'],os.path.join(save_path,'D_F.pkl'))
    torch.save(state['D_J'], os.path.join(save_path,'D_J.pkl'))
    torch.save(state['D'],os.path.join(save_path,'D.pkl'))
    torch.save(state, os.path.join(save_path, 'model.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(save_path, 'model.pth.tar'), os.path.join(save_path,'model_best.pth.tar'))
        shutil.copyfile(os.path.join(save_path, 'E_G.pkl'), os.path.join(save_path,'E_G_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'E_F.pkl'), os.path.join(save_path,'E_F_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'D_G.pkl'), os.path.join(save_path,'D_G_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'D_F.pkl'), os.path.join(save_path,'D_F_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'D_J.pkl'), os.path.join(save_path,'D_J_best.pkl'))
        shutil.copyfile(os.path.join(save_path, 'D.pkl'), os.path.join(save_path,'D_best.pkl'))

class AverageMeter(object):
    '''
    Computes and stores the average and current value
    '''
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def choose_N_A_data(image, label, batch_size):
    data_dict = {}
    N_num = label.sum().item()
    if N_num == batch_size:
        data_dict['N'] = image
        data_dict['A'] = None
        return data_dict, N_num
    elif N_num == 0:
        data_dict['N'] = None
        data_dict['A'] = image
        return data_dict, N_num
    else:
        N_data = torch.zeros((N_num, image.size()[1], image.size()[2], image.size()[3]))
        A_data = torch.zeros((batch_size-N_num, image.size()[1], image.size()[2], image.size()[3]))
        try:
            N_index = torch.where(label==1)[0].cpu().numpy().tolist()
            A_index = torch.where(label==0)[0].cpu().numpy().tolist()
        except:
            N_index = torch.where(label==1)[0].numpy().tolist()
            A_index = torch.where(label==0)[0].numpy().tolist()
        for idx, i in enumerate(N_index):
            N_data[idx,:,:,:] = image[i,:,:,:]
        for idx, i in enumerate(A_index):
            A_data[idx,:,:,:] = image[i,:,:,:]
        data_dict['N'] = N_data
        data_dict['A'] = A_data
    return data_dict, N_num