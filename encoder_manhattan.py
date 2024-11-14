class ManhattanEncoder(LinearEncoder):
    def __init__(self, dim=10000, num=256, r=2):
        super().__init__(dim=dim, num=num)
        self.r = r
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.item_mem = None

    def compute_modular_distance(self, u, v):
        """Compute the modular Manhattan distance between two vectors."""
        diff1 = torch.remainder(u - v, self.r)
        diff2 = torch.remainder(v - u, self.r)
        delta = torch.minimum(diff1, diff2)
        return torch.sum(delta)

    def group_bundle(self, lst):
        """
        Bundle a list of vectors using modular arithmetic.
        
        Args:
            lst (Tensor): A list of vectors to bundle, shape (num_vectors, dim).
        
        Returns:
            Tensor: A single vector representing the bundled result, shape (dim,).
        """
        # Step 1: Sum all vectors element-wise
        summed_vector = torch.sum(lst, dim=0)
        
        # Step 2: Apply modular arithmetic to ensure all elements are within the range [0, r)
        bundled_vector = torch.remainder(summed_vector, self.r)
        
        return bundled_vector

    def encode_one_img(self, x):
        """Encoding for a single image with modular arithmetic."""
        rv = self.item_mem[x[0]]
        for i in range(1, x.shape[0]):
            rv = torch.roll(rv, i)
            rv = torch.remainder(-rv * self.item_mem[x[i]], self.r)
        return rv

    def encode_data_extract_labels(self, datast):
        n = len(datast)
        rv = torch.zeros((n, self.dim)).to(self.device)
        labels = torch.zeros(n).long().to(self.device)
        print('start encoding data with Manhattan distance...')

        batch_size = 128
        data_loader = torch.utils.data.DataLoader(datast, batch_size=batch_size, shuffle=False)

        start_idx = 0
        for batch in data_loader:
            imgs, batch_labels = batch
            imgs = imgs.to(self.device)
            encoded_batch, _ = self.encode_data_extract_labels_batch(imgs)

            batch_size = imgs.size(0)
            rv[start_idx:start_idx + batch_size] = encoded_batch
            labels[start_idx:start_idx + batch_size] = batch_labels

            start_idx += batch_size
            if start_idx % 1000 == 0:
                print(f"{start_idx} images encoded")

        print('finish encoding data with Manhattan distance')
        return rv, labels

    def similarity(self, x, y):
        """
        Compute similarity between vectors using modular Manhattan distance.
        Lower distance means higher similarity.
        """
        n = x.size(-1)
        distance = self.compute_modular_distance(x, y)
        scaled_distance = distance * 4 / (n * self.r)
        return 1 - scaled_distance
