import torch
from models import discriminator, encoder_decoder
import os
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

def get_sample(dataset, index):
    out = torch.zeros((1,1,224,224))
    image, label = dataset[index]
    fname = dataset.imgs[index][0].split('/')[-1]
    out[0,0,:,:] = image[0,:,:]
    return out, fname


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    DATA_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/data'
    LOG_PATH = '/home/ps/disk12t/ACY/AD/MIA/data/pred'
    CHECKPOINT_PATH = '/home/ps/disk12t/ACY/AD/MIA/log/220515_1/model.pth.tar'
    z_p_path = '/home/ps/disk12t/ACY/AD/MIA/data/pred/z_p'
    z_pp_path = '/home/ps/disk12t/ACY/AD/MIA/data/pred/z_pp'
    y_p_path = '/home/ps/disk12t/ACY/AD/MIA/data/pred/y_p'
    a_path = '/home/ps/disk12t/ACY/AD/MIA/data/pred/a'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # dataset
    eye_dataset = ImageFolder(root=DATA_PATH, transform=transform)

    checkpoint = torch.load(CHECKPOINT_PATH)


    # models
    E_G = encoder_decoder.Encoder(norm='in')
    E_F = encoder_decoder.Encoder(norm='none')
    D_G = encoder_decoder.Decoder()
    D_F = encoder_decoder.Decoder()
    D_J = encoder_decoder.Decoder_j()
    D = discriminator.MultiscaleDiscriminator(1)

    E_G.load_state_dict(checkpoint['E_G'])
    E_F.load_state_dict(checkpoint['E_F'])
    D_G.load_state_dict(checkpoint['D_G'])
    D_F.load_state_dict(checkpoint['D_F'])
    D_J.load_state_dict(checkpoint['D_J'])

    E_G.to(device)
    E_F.to(device)
    D_G.to(device)
    D_F.to(device)
    D_J.to(device)
    D.to(device)

    total_num = len(eye_dataset.imgs)
    for i in tqdm(range(total_num)):
        image, fname = get_sample(eye_dataset, i)
        image = (image - 0.5) * 2
        image = image.to(device)
        c_z = E_G(image)
        y_p = D_G(c_z)
        c_s = E_F(image)
        a = D_F(c_s)
        z_pp = D_J(c_z, c_s)
        z_p = y_p + a
        save_image(z_pp / 2 + 0.5, os.path.join(z_p_path, fname))
        save_image(z_p / 2 + 0.5, os.path.join(z_pp_path, fname))
        save_image(y_p / 2 + 0.5, os.path.join(y_p_path, fname))
        save_image(a, os.path.join(a_path, fname))