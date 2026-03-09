from torchvision.datasets import MNIST
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),      
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,))  
])


mnist_train = MNIST(r"C:\data_sets",train=True,download=True,transform=transform)
mnist_test = MNIST(r"C:\data_sets",train=False,download=True,transform=transform)

if __name__ == "__main__":
    x,y = next(iter(mnist_train))
    print(x,y)