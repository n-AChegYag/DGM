from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':

    data_path = '/home/ps/disk12t/ACY/AD/MIA/data/eye'
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    eye_dataset = ImageFolder(root=data_path, transform=transform)
    eye_dataloader = DataLoader(eye_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(eye_dataset[1])