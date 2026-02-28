from torchvision.datasets import MNIST
import torchvision.transforms as transform

transforms = transform.Compose([transform.ToTensor()])


train_data = MNIST(r"C:\data_sets",train=True,download=True,transform=transforms)
test_data = MNIST(r"C:\data_sets",train=False,download=True,transform=transforms)

if __name__ == "__main__":
    x,y = next(iter(train_data))
    print(x,y)