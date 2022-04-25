# train_step

total_iter += 1

# def training_step(data_iter, network, loss_fn, optimizer, args):
#     input, target = data_iter.next()
#     if args.cuda:
#         input = input.to(args.cuda_device)
#         target = target.to(args.cuda_device)

#     pred = network(input)
#     loss = loss_fn(pred, target)

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

class Engine():
    def __ini__(self, args, network, loss_fn, optimizer):
        self.args = args
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        if args.train:
            self.network.train()
            shuffle = True
            phase = "train"
            data_transform = transforms.Compose([
                transforms.ToTensor(),  # divides float version by 255
            ])
            self.description = "[Training]"
        else:
            self.network.eval()
            shuffle = False
            phase = 'val'
            data_transform = transforms.Compose([
                transforms.ToTensor(),  # divides float version by 255
                # transforms.CenterCrop(opt.image_size),
            ])
            self.description = "[Testing]"

        self.set_dataloader = DatasetCityscapes(opt=args, phase=phase, data_transform=data_transform)
        self.dataloader = torch.utils.data.DataLoader(set_dataloader, batch_size=args.batch_size, shuffle=shuffle,
                                                           num_workers=args.n_threads, drop_last=True)
        print("Dataset size : ", len(self.dataloader))
        self.progress_bar = tqdm.tqdm(range(len(self.dataloader)))
        self.total_iter = 0
        self.global_cm = 0
        self.accuracy_metric = AccuracyMetric(global_cm=global_cm)

    def load_checkpoint(self):
        pass
    def train(self):
        pass
    def evaluate(self):
        pass