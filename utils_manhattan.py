import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from encoder_manhattan import LinearEncoder, RandomFourierEncoder, ManhattanEncoder

def quantize(data, precision=8):
    scaling_factor = 2 ** (precision - 1) - 1
    data = np.round(data * scaling_factor)
    return (data + scaling_factor) / 255.0

class HDDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load(args):
    print("Loading encoded training data...")
    train_hd = torch.load(f'{args.data_dir}/train_hd.pt')
    y_train = torch.load(f'{args.data_dir}/y_train.pt')

    print("Loading encoded test data...")
    test_hd = torch.load(f'{args.data_dir}/test_hd.pt')
    y_test = torch.load(f'{args.data_dir}/y_test.pt')

    print(f"Size of encoded training data {train_hd.size()} and test data {test_hd.size()}")
    return train_hd, y_train, test_hd, y_test

def prepare_data(args):
    train_hd, y_train, test_hd, y_test = load(args)
    train_dataset = HDDataset(train_hd, y_train)
    test_dataset = HDDataset(test_hd, y_test)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=16,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=1)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=16,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=1)
    return trainloader, testloader

def encode_and_save(args):
    # Set up data loading with transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load appropriate dataset with progress bar
    print("Loading dataset...")
    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root=args.raw_data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=args.raw_data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'fmnist':
        trainset = torchvision.datasets.FashionMNIST(root=args.raw_data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=args.raw_data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'cifar':
        trainset = torchvision.datasets.CIFAR10(root=args.raw_data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=args.raw_data_dir, train=False, download=True, transform=transform)
    elif args.dataset == "isolet":
        import pickle
        def dataset(source):
            with open(source, 'rb') as f:
                isolet = pickle.load(f)
            trainData, trainLabels, testData, testLabels = isolet
            return np.array(trainData), np.array(trainLabels), np.array(testData), np.array(testLabels)

        x_train, y_train, x_test, y_test = dataset(source=f'./{args.raw_data_dir}/isolet/isolet.pkl')
        x_train, y_train = torch.tensor(quantize(x_train, precision=8)).unsqueeze(1), torch.tensor(y_train).long()
        x_test, y_test = torch.tensor(quantize(x_test, precision=8)).unsqueeze(1), torch.tensor(y_test).long()
        trainset = HDDataset(x_train, y_train)
        testset = HDDataset(x_test, y_test)
    elif args.dataset == "ucihar":
        x_train_path = f'./{args.raw_data_dir}/ucihar/train/x_train.txt'
        y_train_path = f'./{args.raw_data_dir}/ucihar/train/y_train.txt'
        x_test_path = f'./{args.raw_data_dir}/ucihar/test/x_test.txt'
        y_test_path = f'./{args.raw_data_dir}/ucihar/test/y_test.txt'

        def load_data(feature_file_path, label_file_path):
            x_train = open(feature_file_path, 'r')
            x_train = x_train.readlines()
            for idx in range(len(x_train)):
                x_train[idx] = x_train[idx].split()
            x_train = np.array(x_train, dtype=np.float32)
            y_train = open(label_file_path, 'r')
            y_train = y_train.readlines()
            y_train = np.array(y_train, dtype=np.int32) - 1
            return x_train, y_train

        x_train, y_train = load_data(x_train_path, y_train_path)
        x_test, y_test = load_data(x_test_path, y_test_path)
        x_train, y_train = torch.tensor(quantize(x_train, precision=8)).unsqueeze(1), torch.tensor(y_train).long()
        x_test, y_test = torch.tensor(quantize(x_test, precision=8)).unsqueeze(1), torch.tensor(y_test).long()
        trainset = HDDataset(x_train, y_train)
        testset = HDDataset(x_test, y_test)
    else:
        raise ValueError("Dataset is not supported.")

    assert len(trainset[0][0].size()) > 1
    channels = trainset[0][0].size(0)
    print('# of channels of data', channels)
    input_dim = torch.prod(torch.tensor(list(trainset[0][0].size())))
    print('# of training samples and test samples', len(trainset), len(testset))

    # Initialize encoder
    if args.model == 'linear-hdc':
        print("Encoding to binary HDC with linear hamming distance.")
        encoder = LinearEncoder(dim=args.dim)
    elif args.model == 'manhattan-hdc':
        print("Encoding to HDC with Manhattan distance.")
        encoder = ManhattanEncoder(dim=args.dim, num=256, r=args.r)
    elif 'rff' in args.model:
        print("Encoding with random fourier features encoder.")
        encoder = RandomFourierEncoder(input_dim=input_dim, gamma=args.gamma, gorder=args.gorder, output_dim=args.dim)
    else:
        raise ValueError("No such feature type is supported.")

    # Create dataloaders with batching for encoding
    batch_size = 128  # Adjust based on your memory constraints
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build and save item memory
    print("Building item memory...")
    mem = encoder.build_item_mem()
    print(f"Encoded pixels to hypervectors with size: {mem.size()}")
    torch.save(mem, f'{args.data_dir}/item_mem.pt')

    # Encoding training data
    print("Encoding training data...")
    with tqdm(total=len(trainset), desc="Training Data Encoding") as pbar:
        train_hd, y_train = [], []
        for item in trainset:
            encoded, label = encoder.encode_data_extract_labels([item])  # Process one item at a time
            train_hd.append(encoded)
            y_train.append(label)
            pbar.update(1)  # Update the progress bar after each item
    
    train_hd = torch.stack(train_hd)  # Combine list of tensors if needed
    y_train = torch.tensor(y_train)  # Adjust as per your label data type
    torch.save(train_hd, f'{args.data_dir}/train_hd.pt')
    torch.save(y_train, f'{args.data_dir}/y_train.pt')
    del train_hd, y_train
    torch.cuda.empty_cache()  # in case of CUDA OOM
    
    # Encoding test data
    print("Encoding test data...")
    with tqdm(total=len(testset), desc="Test Data Encoding") as pbar:
        test_hd, y_test = [], []
        for item in testset:
            encoded, label = encoder.encode_data_extract_labels([item])  # Process one item at a time
            test_hd.append(encoded)
            y_test.append(label)
            pbar.update(1)  # Update the progress bar after each item

    test_hd = torch.stack(test_hd)  # Combine list of tensors if needed
    y_test = torch.tensor(y_test)  # Adjust as per your label data type
    torch.save(test_hd, f'{args.data_dir}/test_hd.pt')
    torch.save(y_test, f'{args.data_dir}/y_test.pt')
    del test_hd, y_test
    torch.cuda.empty_cache()
