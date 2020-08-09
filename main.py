# from classifier import AudioSileroMFCC, MFCCSimpleAudioNet, create_data_loaders
import classifier
import trainer
# import torch.optim as optim
import torch


def main():
    train_loader, valid_loader = classifier.create_data_loaders('train.csv', batch_size=50,
                                                                dataset_class=classifier.AudioSileroMFCC)

    net = classifier.MFCCSimpleAudioNet()
    net.to('cpu')
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    trainer.train(net, optimizer, torch.nn.CrossEntropyLoss(), train_loader, valid_loader, epochs=20)


if __name__ == "__main__":
    main()
