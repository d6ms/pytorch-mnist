import os
from random import randint
from argparse import ArgumentParser

from torch import no_grad, save, load
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from model import MnistModel


def dataloader(batch_size=100, num_workers=2):
    transform = Compose([ToTensor(), Normalize((0.5, ), (0.5, ))])
    trainset = MNIST(root='data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testset = MNIST(root='data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return trainloader, testloader


def train(save_path='models/mnist.model'):
    # parameters
    epochs = 5
    learning_rate = 0.001

    # settings
    model = MnistModel()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    # train and evaluate
    history = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    trainloader, testloader = dataloader()
    for epoch in range(epochs):
        model.train()  # train mode
        running_loss = 0
        for i, (data, labels) in enumerate(trainloader):
            data = data.view(-1, 28 * 28)  # 入力を1次元に変換
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()  # 重みの更新

            # print statistics (after every 100 minibatches)
            running_loss += loss.item()
            if i % 100 == 99:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(epoch + 1, i + 1, running_loss / 100))
                running_loss = 0
        history['train_loss'].append(loss)

        # evaluate
        model.eval()  # eval mode
        test_loss, correct_cnt = 0, 0
        with no_grad():  # 自動微分をoffにしてメモリ節約
            for data, labels in testloader:
                data = data.view(-1, 28 * 28)  # 入力を1次元に変換
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                correct_cnt += prediction.eq(labels.view_as(prediction)).sum().item()
                test_loss += loss_fn(output, labels).item()
        n_samples = len(testloader.dataset)
        test_loss, test_acc = test_loss / n_samples, correct_cnt / n_samples
        print(f'Test loss (avg): {test_loss}, Accuracy: {test_acc}')
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
    
    # save model
    dir = save_path[:save_path.rindex('/')]
    if len(dir) > 0 and not os.path.exists(dir):
        os.makedirs(dir)
    save(model.state_dict(), save_path)


def predict(model_path='models/mnist.model'):
    _, loader = dataloader()
    data, label = loader.dataset[randint(0, len(loader.dataset) - 1)]
    data = data.view(-1, 28 * 28)


    model = MnistModel()
    model.load_state_dict(load(model_path))
    output = model(data)
    prediction = output.argmax(dim=1, keepdim=True)

    print(f'prediction: {prediction.item()}, label: {label}')


if __name__ == '__main__':
    usage = f'Usage: python {__file__} [-t | --train] [-p | --predict]'
    parser = ArgumentParser(usage=usage)
    parser.add_argument('-t', '--train', action='store_true', help='train mnist model')
    parser.add_argument('-p', '--predict', action='store_true', help='load model and demonstrate prediction')
    args = parser.parse_args()
    if args.train:
        train()
    elif args.predict:
        predict()
    else:
        parser.print_help()
