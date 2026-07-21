from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((128, 128)),      
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,),(0.5))  
])


train = CIFAR10(r"/mnt/data2/datasets/cifar10",train=True,download=True,transform=transform)
test = CIFAR10(r"/mnt/data2/datasets/cifar10",train=False,download=True,transform=transform)

if __name__ == "__main__":
    x,y = next(iter(train))
    print(x,y)