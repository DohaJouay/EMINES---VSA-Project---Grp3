#!/usr/bin/env python
# coding: utf-8
import torch
import torchvision
import numpy as np
import time
from torchvision import transforms
import matplotlib.pyplot as plt
from encoder_manhattan import LinearEncoder, RandomFourierEncoder, ManhattanEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

def quantize(data, precision=8):
    scaling_factor = 2 ** (precision - 1) - 1
    data = np.round(data * scaling_factor)
    return (data + scaling_factor) / 255.0

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
    elif args.dataset in ["isolet", "ucihar"]:
        trainset, testset = load_custom_dataset(args)
    else:
        raise ValueError("Dataset is not supported.")

    # Create dataloaders with batching
    batch_size = 128  # Adjust based on your memory constraints
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Calculate input dimensions
    channels = trainset[0][0].size(0)
    input_dim = torch.prod(torch.tensor(list(trainset[0][0].size())))
    print(f'# of channels: {channels}, Input dim: {input_dim}')
    print(f'# of training samples: {len(trainset)}, test samples: {len(testset)}')

    # Initialize encoder
    encoder = initialize_encoder(args, input_dim)
    
    # Build and save item memory
    print("Building item memory...")
    mem = encoder.build_item_mem()
    print(f"Encoded pixels to hypervectors with size: {mem.size()}")
    torch.save(mem, f'{args.data_dir}/item_mem.pt')

    # Encode training data in batches
    print("Encoding training data...")
    train_hd, y_train = encode_dataset_in_batches(train_loader, encoder)
    torch.save(train_hd, f'{args.data_dir}/train_hd.pt')
    torch.save(y_train, f'{args.data_dir}/y_train.pt')
    del train_hd, y_train
    torch.cuda.empty_cache()

    # Encode test data in batches
    print("Encoding test data...")
    test_hd, y_test = encode_dataset_in_batches(test_loader, encoder)
    torch.save(test_hd, f'{args.data_dir}/test_hd.pt')
    torch.save(y_test, f'{args.data_dir}/y_test.pt')
    del test_hd, y_test
    torch.cuda.empty_cache()

def initialize_encoder(args, input_dim):
    if args.model == 'linear-hdc':
        print("Encoding to binary HDC with linear hamming distance.")
        return LinearEncoder(dim=args.dim)
    elif args.model == 'manhattan-hdc':
        print("Encoding to HDC with Manhattan distance.")
        return ManhattanEncoder(dim=args.dim, num=256, r=args.r)
    elif 'rff' in args.model:
        print("Encoding with random fourier features encoder.")
        return RandomFourierEncoder(input_dim=input_dim, gamma=args.gamma, 
                                  gorder=args.gorder, output_dim=args.dim)
    else:
        raise ValueError("No such feature type is supported.")

def encode_dataset_in_batches(loader, encoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoded_data = []
    labels = []
    
    with tqdm(total=len(loader), desc="Encoding batches") as pbar:
        for batch, target in loader:
            if torch.cuda.is_available():
                batch = batch.to(device)
            
            # Encode batch
            encoded_batch = encoder.encode_data_extract_labels_batch(batch)
            encoded_data.append(encoded_batch[0].cpu())
            labels.extend(target.tolist())
            
            pbar.update(1)
            torch.cuda.empty_cache()  # Clear GPU memory after each batch
    
    # Combine all batches
    encoded_tensor = torch.cat(encoded_data, dim=0)
    labels_tensor = torch.tensor(labels)
    
    return encoded_tensor, labels_tensor

def load_custom_dataset(args):
    if args.dataset == "isolet":
        import pickle
        with open(f'./{args.raw_data_dir}/isolet/isolet.pkl', 'rb') as f:
            isolet = pickle.load(f)
        trainData, trainLabels, testData, testLabels = isolet
        x_train = torch.tensor(quantize(trainData, precision=8)).unsqueeze(1)
        y_train = torch.tensor(trainLabels).long()
        x_test = torch.tensor(quantize(testData, precision=8)).unsqueeze(1)
        y_test = torch.tensor(testLabels).long()
    elif args.dataset == "ucihar":
        x_train, y_train = load_ucihar_data(args, "train")
        x_test, y_test = load_ucihar_data(args, "test")
    
    return HDDataset(x_train, y_train), HDDataset(x_test, y_test)

def load_ucihar_data(args, split):
    base_path = f'./{args.raw_data_dir}/ucihar/{split}/'
    x_path = base_path + f'x_{split}.txt'
    y_path = base_path + f'y_{split}.txt'
    
    # Load features
    with open(x_path, 'r') as f:
        x_data = np.array([line.split() for line in f.readlines()], dtype=np.float32)
    
    # Load labels
    with open(y_path, 'r') as f:
        y_data = np.array(f.readlines(), dtype=np.int32) - 1
    
    # Convert to tensors
    x_tensor = torch.tensor(quantize(x_data, precision=8)).unsqueeze(1)
    y_tensor = torch.tensor(y_data).long()
    
    return x_tensor, y_tensor
