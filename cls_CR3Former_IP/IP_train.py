import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from operator import truediv
import CR3FormerDemo.cls_CR3Former_IP.get_cls_map
import time
import CR3FormerDemo.cls_CR3Former_IP.CR3Former
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


def loadData():
    data = sio.loadmat('..\data\img2.mat')['img2']
    labels = sio.loadmat('..\data\Indian_pines_gt.mat')['indian_pines_gt']

    return data, labels

def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX

def padWithZeros(X, margin=2):

    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):

    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1

    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test

# BATCH_SIZE_TRAIN = 40 #hc
BATCH_SIZE_TRAIN = 64 #ip
# BATCH_SIZE_TRAIN = 40 #sa
# BATCH_SIZE_TRAIN = 64 #LK
# BATCH_SIZE_TRAIN = 128 #UH
# BATCH_SIZE_TRAIN = 128 #pu
# BATCH_SIZE_TRAIN = 192 #td
# BATCH_SIZE_TRAIN = 128 #fc

def create_data_loader():
    class_num = 16
    # class_num = 15
    # class_num = 9
    # class_num = 6
    # class_num = 20
    X, y = loadData()
    # test_ratio = 0.99  # lk
    # test_ratio = 0.99 #fc
    # test_ratio = 0.99  # hc
    test_ratio = 0.95 # ip
    patch_size = 11 #ip
    # patch_size = 11 # pu
    # patch_size = 11 #uh
    # patch_size = 7 #lk
    # patch_size = 19 #hc
    # patch_size = 31 #sa
    # patch_size = 15  # td
    # patch_size = 11  # fc
    pca_components = 30

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    # high_pca, high_y_all = createImageCubes(high_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    X = X.transpose(0, 4, 3, 1, 2)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)
  
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=False,
                                               num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                batch_size=BATCH_SIZE_TRAIN,
                                                shuffle=False,
                                                num_workers=0,
                                              )

    return train_loader, test_loader, all_data_loader, y

""" Training dataset"""

class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain):

        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]
    def __len__(self):

        return self.len

""" Testing dataset"""

class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):

        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, labels):
        ce = nn.CrossEntropyLoss(weight=self.weight, reduction="none")(logits, labels)
        p = torch.exp(-ce)
        loss = (1 - p) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

def train(train_loader, epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = CR3FormerDemo.cls_CR3Former_IP.CR3Former.SSFTTnet().to(device)
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    total_loss = 0
    for epoch in range(epochs):
        net.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print('[Epoch: %d]   [loss avg: %.4f]   [current loss: %.4f]' % (epoch + 1,
                                                                         total_loss / (epoch + 1),
                                                                         loss.item()))

    print('Finished Training')

    return net, device

def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def AA_andEachClassAccuracy(confusion_matrix):

    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):

    target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
        , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                    'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                    'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                    'Stone-Steel-Towers']
    # target_names = ['Corn','Cotton','Sesame','Broad-leaf soybean', 'Narrow-leaf soybean','Rice','Water',
    #                     'Roads and houses','Mixed weed']
    # target_names = ['Crops 1','Bare Soil','Shrubs','Playground track','Road','Trees','Crops 2',
    #                 'Shadow','Concrete floor','Concrete roof','Caigang watt roof','Cars',
    #                 'Asphalt roof','Grass','Glazed tile roof','Clay tile roof','Special material',
    #                 'Piazza area','White tile floor','Black tile floor']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

class ActivationOutputData():
    # Network outputs
    outputs = None
    def __init__(self, layer):
        # Register the callback function on the (layer_num) layer of the model and pass in the processing function.
        self.hook = layer.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, inputs, outputs):
        self.outputs = outputs.cpu()
    def remove(self):
        # Called by a callback handle, used to remove the callback function from the network layer.
        self.hook.remove()

if __name__ == '__main__':

    train_loader, test_loader, all_data_loader, y_all= create_data_loader()
    tic1 = time.perf_counter()
    net, device = train(train_loader, epochs=75)  #ip
    # net, device = train(train_loader, epochs=100) #sa
    # net, device = train(train_loader, epochs=75) #lk
    # net, device = train(train_loader, epochs=100) #uh hc pu
    # net, device = train(train_loader, epochs=50) #hc
    # net, device = train(train_loader, epochs=150) #fc
    # net, device = train(train_loader, epochs=75)  # td
    torch.save(net.state_dict(), 'cls_params/CR3Former_params.pth')
    toc1 = time.perf_counter()
    tic2 = time.perf_counter()
    y_pred_test, y_test = test(device, net, test_loader)
    toc2 = time.perf_counter()
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    classification = str(classification)
    Training_Time = toc1 - tic1
    Test_time = toc2 - tic2
    file_name = "cls_result/classification_report.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(kappa))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(oa))
        x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(aa))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(each_acc))
        x_file.write('\n')
        x_file.write('{}'.format(classification))
        x_file.write('\n')
        x_file.write('{}'.format(confusion))

    CR3FormerDemo.cls_CR3Former_IP.get_cls_map.get_cls_map(net, device, all_data_loader, y_all)


    # %% md

    ## t-SNE Visualization

    # %%

    # def scatter(x, colors, num_classes=9):
    #     # palette = np.array([[192, 192, 192], [0, 255, 1], [0, 255, 255], [0, 128, 1], [255, 0, 254],
    #     #                     [165, 82, 40], [129, 0, 127], [255, 0, 0], [255, 255, 0]]) / 255.
    #     palette = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    #                         [255, 0, 255], [0, 255, 255], [200, 100, 0], [0, 200, 100],
    #                         [100, 0, 200],[200, 0, 100],[100, 200, 0],[0, 100, 200],[150, 75, 75],
    #                         [75, 150, 75],[75, 75, 150],[255, 100, 100]]) / 255.
    #
    #     # We create a scatter plot.
    #     fig = plt.figure(figsize=(8, 8))
    #     ax = plt.subplot(aspect='equal')
    #     sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int_)])
    #     plt.xlim(-25, 25)
    #     plt.ylim(-25, 25)
    #     ax.axis('off')
    #     ax.axis('tight')
    #
    #     # We add the labels for each digit.
    #     txts = []
    #     for i in range(num_classes):
    #         # Position of each label.
    #         xtext, ytext = np.median(x[colors == i, :], axis=0)
    #         txt = ax.text(xtext, ytext, str(i + 1), fontsize=24)
    #         txt.set_path_effects([PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()])
    #         txts.append(txt)
    #     fig.savefig('./tsne_OUR.png', bbox_inches='tight')


    # %%

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # test_codings = []
    # test_tar = []
    # for batch_idx, (image, label) in enumerate(test_loader):
    #     with torch.no_grad():
    #         image = image.to(device)
    #         label = label.to(device)
    #         final_features = ActivationOutputData(net)  # (1, 400, 8)
    #         _ = net(image)
    #         final_features.remove()
    #     test_codings.append(final_features.outputs.cpu().data)
    #     test_tar.append(label.cpu().data)
    #
    # x_test = torch.concat(test_codings)
    # y_test = torch.concat(test_tar)
    #
    # x_test_numpy = x_test.numpy()
    # y_test_numpy = y_test.numpy()
    #
    # # %%
    #
    # tsne_proj = TSNE(random_state=42).fit_transform(x_test_numpy)
    #
    # scatter(tsne_proj, y_test_numpy)






