from torchvision.datasets import MNIST
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),      
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,))  
])


train_data = MNIST(r"C:\data_sets",train=True,download=True,transform=transform)
test_data = MNIST(r"C:\data_sets",train=False,download=True,transform=transform)

if __name__ == "__main__":
    x,y = next(iter(train_data))
    print(x,y)