import torch
import torch.utils.data as data
from torchvision import transforms, utils
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import logging


import metrics


transformer = {'Train':transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomRotation(180),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.ToTensor(),
]),

'Gray':transforms.Compose([
    transforms.Resize((336,336)),
    # transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]),
'ColorAll':transforms.Compose([
    transforms.Resize((336,336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]),
'BinaryFluid':transforms.Compose([
    transforms.Resize((336,336)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]),
'SegTrain':transforms.Compose([
    transforms.Resize((336,336)),
    transforms.Grayscale(1)
]),
'SegTest':transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.Grayscale(1)
])
}




# 用RF打分做特征排序，返回排序后的特征字段列表
def fea_sel(X_train, y_label):
    data_np = np.array(X_train)
    list_name = list(X_train.columns)

    X = preprocessing.scale(np.array([line for line in data_np]))
    y = np.array([line for line in y_label])

    clf = RandomForestClassifier(
        n_estimators=500, max_depth=None, random_state=256)
    clf.fit(X, y)
    rt_import = np.array(clf.feature_importances_)
    rt_import = rt_import.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    rt_import_scale = min_max_scaler.fit_transform(rt_import)
    dict_import = zip(list_name, rt_import_scale)
    sort_dict = dict((names, value) for names, value in dict_import)
    fea_score = dict(
        sorted(sort_dict.items(), key=lambda asd: asd[1], reverse=True))
    fea_name = list(fea_score.keys())
    return fea_name


# 为data表获取斑块类型标签  
def get_label(data,ref_data_root):
    ref_data = pd.read_csv(ref_data_root,index_col=2)
    for ID in data.index:
        data.loc[ID, 'Type'] = ref_data.loc[ID, 'TypeOfPlaque']
        


class MTLDataset(data.Dataset):
    # auxiliary: optional propertiy: 'US','MIX'
    # filter: input can be any 1D list object, or specified methods ---- 'rf_importance' 
    def __init__(self,img_root,my_transformer=transformer,ref_data_root = '../ResearchData/data_ultrasound_-1.csv',num_classes = 4,auxiliary = 'US',us_path=None,mix_path=None,screener='rf_importance',screen_num = 10, train_or_test = 'Test'): #TODO
        self.transformer = my_transformer
        self.datatype = auxiliary
        self.num_classes = num_classes
        self.train_or_test = train_or_test
        if(auxiliary == 'US' and us_path):
            logging.debug('US Dataset has prepared to init.')
            data = pd.read_csv(us_path,index_col=-1)
        elif(auxiliary == 'MIX' and mix_path):
            logging.debug('MIX Dataset has prepared to init.')
            data = pd.read_csv(mix_path,index_col=-1)
        else:
            raise Exception('Dataset Function Parameters are not avaiable, please check your Function.')
        # 图片文件列表
        file_list = os.listdir(img_root)
        # 过滤出所有jpg文件
        img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', file_list))
        self.img_path_list = [img_root + filename for filename in img_list]
        # 获取图片名列表，图片名即病历ID
        self.img_ID_list = [name[:-4] for name in img_list]            
        get_label(data, ref_data_root)
        if (num_classes == 4):
            label = np.array([i-1 for i in data['Type']])
        elif (num_classes == 2):
            label = np.array([1 if (i == 2 or i == 3) else 0 for i in data['Type']])
        elif (num_classes == 3):
            raise Exception('Undone.')
        else:
            raise Exception('Check parameter num_classes.')
        
        self.data = self.screen(data, label, screener, screen_num = screen_num)
        
        
    def __getitem__(self, index):
        with Image.open(self.img_path_list[index]) as pil_img:
            if(self.train_or_test == 'Train'):
                img = self.transformer['Gray'](self.transformer['Train'](pil_img))
            elif(self.train_or_test == 'Test'): 
                img = self.transformer['Gray'](pil_img)
        img_ID = self.img_ID_list[index]
        img_ID_data = torch.from_numpy(np.array(self.data.loc[img_ID,:],dtype=np.float32))
        
        numeric_data = img_ID_data[:-1]
        type_label = img_ID_data[-1]
        type_label4 = type_label.long()
        if (type_label == 0 or type_label == 3):
            type_label2 = torch.from_numpy(np.array(0,dtype=np.int64))
        else:
            type_label2 = torch.from_numpy(np.array(1,dtype=np.int64))
        # type_label2 = type_label2.long()
        # logging.error(img.shape,numeric_data.shape,type_label)
        return img, numeric_data, type_label4, type_label2

    def __len__(self):
        return len(self.img_path_list)

    # 筛选前10个特征
    def screen(self, data, label, method, screen_num = 10):
        data.drop(['Type'],axis = 1, inplace=True)
        if(isinstance(method,str)):
            if (method == 'rf_importance'):
                fea_name = fea_sel(data,label)[:screen_num]
                data = data.loc[:, fea_name]
                logging.info('Fea_name {} by rf_importance: {}'.format(screen_num,fea_name) )

        elif (isinstance(method, list)):
            # method should be a list of fea_name.
            fea_name = method[:screen_num]
            data = data.loc[:, fea_name]
            logging.info('Fea_name by list: {}'.format(fea_name))
            
        else:
            raise Exception('WARNING from SHYyyyyyy: Check your Screener Configuration.')
        data['Type'] = label
        return data






class SegDataset(data.Dataset):
    # auxiliary: optional propertiy: 'US','MIX'
    # filter: input can be any 1D list object, or specified methods ---- 'rf_importance' 
    def __init__(self,img_root,seg_label_root,fluid_root=None, my_transformer = transformer,ref_data_root = '../ResearchData/data_ultrasound_-1.csv',num_classes = 4,auxiliary = 'US',us_path=None,mix_path=None,screener='rf_importance',screen_num = 10, train_or_test = 'Test'): #TODO
        self.transformer = my_transformer
        self.datatype = auxiliary
        self.num_classes = num_classes
        self.train_or_test = train_or_test
        if(auxiliary == 'US' and us_path):
            # logging.debug('US Dataset has prepared to init.')
            data = pd.read_csv(us_path,index_col=-1)
        elif(auxiliary == 'MIX' and mix_path):
            # logging.debug('MIX Dataset has prepared to init.')
            data = pd.read_csv(mix_path,index_col=-1)
        else:
            raise Exception('Dataset Function Parameters are not avaiable, please check your Function.')
        # 图片文件列表
        file_list = os.listdir(img_root)
        # 过滤出所有jpg文件
        img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', file_list))
        # 获取图片名列表，图片名即病历ID
        get_label(data, ref_data_root)
        if(num_classes == 4):
            label = np.array([i-1 for i in data['Type']])
        elif(num_classes == 2):
            label = np.array([1 if (i == 2 or i == 3) else 0 for i in data['Type']])
        elif(num_classes == 3):
            raise Exception('Undone.')
        else:
            raise Exception('Check parameter num_classes.')
        
        data = self.screen(data, label, screener, screen_num=screen_num)
        # 读取分割标签列表，然后按标签给img_path_list做交集（子集）
        seg_file_list = os.listdir(seg_label_root)
        seg_img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', seg_file_list))
        inter_img_list = [filename for filename in img_list if filename in seg_img_list]

        if fluid_root is not None:
            fluid_file_list = os.listdir(fluid_root)
            fluid_img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', fluid_file_list))
            inter_img_list = [filename for filename in inter_img_list if filename in fluid_img_list]

        # 先只做子集的情况下，不必再考虑img_path_list
        self.img_ID_list = [name[:-4] for name in inter_img_list]            
        self.img_path_list = [img_root + filename for filename in inter_img_list]
        self.seg_img_path_list = [seg_label_root + filename for filename in inter_img_list]
        self.data = self.screen(data, label, screener, screen_num = screen_num)
        
    def __getitem__(self, index):
        with Image.open(self.img_path_list[index]) as pil_img:
            if(self.train_or_test == 'Train'):
                img = self.transformer['Gray'](self.transformer['Train'](pil_img))
            elif(self.train_or_test == 'Test'): 
                img = self.transformer['Gray'](pil_img)
        img_ID = self.img_ID_list[index]
        img_ID_data = torch.from_numpy(np.array(self.data.loc[img_ID,:],dtype=np.float32))
        with Image.open(self.seg_img_path_list[index]) as seg_pil_img:
            if (self.train_or_test == 'Train'):
                seg_label = self.transformer['SegTrain'](seg_pil_img)
            elif (self.train_or_test == 'Test'):
                seg_label = self.transformer['SegTest'](seg_pil_img)
            seg_label = np.array(seg_label)
            seg_label[seg_label > 0] = 1
            seg_label[seg_label == 0] = 0
            seg_label = torch.unsqueeze(torch.LongTensor(seg_label), 0)

            # seg_label = torch.unsqueeze(torch.LongTensor(seg_label), 0)
            # seg_label = metrics.one_hot(seg_label, 2)
            # seg_label = torch.squeeze(seg_label, 0)
        numeric_data = img_ID_data[:-1]
        type_label = img_ID_data[-1]
        type_label4 = type_label.long()
        if (type_label == 0 or type_label == 3):
            type_label2 = torch.from_numpy(np.array(0,dtype=np.int64))
        else:
            type_label2 = torch.from_numpy(np.array(1,dtype=np.int64))
        # type_label2 = type_label2.long()
        # logging.error(img.shape,numeric_data.shape,type_label)
        return img_ID, img, seg_label, numeric_data, type_label4, type_label2

    def __len__(self):
        return len(self.img_path_list)

    # 筛选前10个特征
    def screen(self, data, label, method, screen_num = 10):
        data.drop(['Type'],axis = 1, inplace=True)
        if(isinstance(method,str)):
            if (method == 'rf_importance'):
                fea_name = fea_sel(data,label)[:screen_num]
                data = data.loc[:, fea_name]
                # logging.debug('Fea_name {} by rf_importance: {}'.format(screen_num,fea_name) )

        elif (isinstance(method, list)):
            # method should be a list of fea_name.
            fea_name = method[:screen_num]
            data = data.loc[:, fea_name]
            # logging.debug('Fea_name by list: {}'.format(fea_name))

        else:
            raise Exception('WARNING from SHYyyyyyy: Check your Screener Configuration.')
        data['Type'] = label
        return data


class FluidSegDataset(data.Dataset):
    # auxiliary: optional propertiy: 'US','MIX'
    # filter: input can be any 1D list object, or specified methods ---- 'rf_importance'
    def __init__(self, img_root, seg_label_root, fluid_root, binary_fluid=False, my_transformer=transformer,
                 ref_data_root='../ResearchData/data_ultrasound_-1.csv', num_classes=4, auxiliary='US', us_path=None,
                 mix_path=None, screener='rf_importance', screen_num=10, train_or_test='Test'):
        self.transformer = my_transformer
        self.datatype = auxiliary
        self.num_classes = num_classes
        self.train_or_test = train_or_test
        self.binary_fluid = binary_fluid
        if (auxiliary == 'US' and us_path):
            # logging.debug('US Dataset has prepared to init.')
            data = pd.read_csv(us_path, index_col=-1)
        elif (auxiliary == 'MIX' and mix_path):
            # logging.debug('MIX Dataset has prepared to init.')
            data = pd.read_csv(mix_path, index_col=-1)
        else:
            raise Exception('Dataset Function Parameters are not avaiable, please check your Function.')
        # 图片文件列表
        file_list = os.listdir(img_root)
        # 过滤出所有jpg文件
        img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', file_list))
        # 获取图片名列表，图片名即病历ID
        get_label(data, ref_data_root)
        if (num_classes == 4):
            label = np.array([i - 1 for i in data['Type']])
        elif (num_classes == 2):
            label = np.array([1 if (i == 2 or i == 3) else 0 for i in data['Type']])
        elif (num_classes == 3):
            raise Exception('Undone.')
        else:
            raise Exception('Check parameter num_classes.')

        # data = self.screen(data, label, screener, screen_num=screen_num)

        # 读取分割标签列表，然后按标签给img_path_list做交集（子集）
        seg_file_list = os.listdir(seg_label_root)
        seg_img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', seg_file_list))
        fluid_file_list = os.listdir(fluid_root)
        fluid_img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', fluid_file_list))
        # 读取存在标签的图像
        inter_img_list = [filename for filename in img_list if filename in seg_img_list]
        inter_img_list = [filename for filename in inter_img_list if filename in fluid_img_list]


        self.img_ID_list = [name[:-4] for name in inter_img_list]
        self.img_path_list = [img_root + filename for filename in inter_img_list]
        self.seg_img_path_list = [seg_label_root + filename for filename in inter_img_list]
        self.fluid_img_list = [fluid_root + filename for filename in inter_img_list]
        self.data = self.screen(data, label, screener, screen_num=screen_num)


    def __getitem__(self, index):
        with Image.open(self.img_path_list[index]) as pil_img:
            if (self.train_or_test == 'Train'):
                img = self.transformer['Gray'](self.transformer['Train'](pil_img))
            elif (self.train_or_test == 'Test'):
                img = self.transformer['Gray'](pil_img)
        with Image.open(self.fluid_img_list[index]) as f_img:
            if self.binary_fluid:
                fluid_img = self.transformer['BinaryFluid'](f_img)
            elif (self.train_or_test == 'Train'):
                fluid_img = self.transformer['ColorAll'](self.transformer['Train'](f_img))
            elif (self.train_or_test == 'Test'):
                fluid_img = self.transformer['ColorAll'](f_img)
        img_ID = self.img_ID_list[index]
        img_ID_data = torch.from_numpy(np.array(self.data.loc[img_ID, :], dtype=np.float32))
        with Image.open(self.seg_img_path_list[index]) as seg_pil_img:
            if (self.train_or_test == 'Train'):
                seg_label = self.transformer['SegTrain'](seg_pil_img)
            elif (self.train_or_test == 'Test'):
                seg_label = self.transformer['SegTest'](seg_pil_img)
            seg_label = np.array(seg_label)
            seg_label[seg_label > 0] = 1
            seg_label[seg_label == 0] = 0
            seg_label = torch.unsqueeze(torch.LongTensor(seg_label), 0)

            # seg_label = torch.unsqueeze(torch.LongTensor(seg_label), 0)
            # seg_label = metrics.one_hot(seg_label, 2)
            # seg_label = torch.squeeze(seg_label, 0)
        numeric_data = img_ID_data[:-1]
        type_label = img_ID_data[-1]
        type_label4 = type_label.long()
        if (type_label == 0 or type_label == 3):
            type_label2 = torch.from_numpy(np.array(0, dtype=np.int64))
        else:
            type_label2 = torch.from_numpy(np.array(1, dtype=np.int64))


        return img_ID, img, fluid_img, seg_label, numeric_data, type_label4, type_label2

    def __len__(self):
        return len(self.img_path_list)

    # 筛选前10个特征
    def screen(self, data, label, method, screen_num=10):
        data.drop(['Type'], axis=1, inplace=True)
        if (isinstance(method, str)):
            if (method == 'rf_importance'):
                fea_name = fea_sel(data, label)[:screen_num]
                data = data.loc[:, fea_name]
                # logging.debug('Fea_name {} by rf_importance: {}'.format(screen_num,fea_name) )

        elif (isinstance(method, list)):
            # method should be a list of fea_name.
            fea_name = method[:screen_num]
            data = data.loc[:, fea_name]
            # logging.debug('Fea_name by list: {}'.format(fea_name))

        else:
            raise Exception('WARNING from SHYyyyyyy: Check your Screener Configuration.')
        data['Type'] = label
        return data


class FluidSegDatasetPureSeg(data.Dataset):
    # auxiliary: optional propertiy: 'US','MIX'
    # filter: input can be any 1D list object, or specified methods ---- 'rf_importance'
    def __init__(self, img_root, seg_label_root, fluid_root, my_transformer=transformer, train_or_test='Test'):
        self.transformer = my_transformer

        self.train_or_test = train_or_test

        # 图片文件列表
        file_list = os.listdir(img_root)
        # 过滤出所有jpg文件
        img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', file_list))
        # 获取图片名列表，图片名即病历ID


        # data = self.screen(data, label, screener, screen_num=screen_num)

        # 读取分割标签列表，然后按标签给img_path_list做交集（子集）
        seg_file_list = os.listdir(seg_label_root)
        seg_img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', seg_file_list))
        fluid_file_list = os.listdir(fluid_root)
        fluid_img_list = list(filter(lambda file: os.path.splitext(file)[1] == '.jpg', fluid_file_list))
        # 读取存在标签的图像

        inter_img_list = [filename for filename in img_list if filename in fluid_img_list]
        inter_img_list = [filename for filename in inter_img_list if filename in seg_img_list]
        self.img_ID_list = [name[:-4] for name in inter_img_list]
        self.img_path_list = [img_root + filename for filename in inter_img_list]
        self.seg_img_path_list = [seg_label_root + filename for filename in inter_img_list]
        self.fluid_img_list = [fluid_root + filename for filename in inter_img_list]


    def __getitem__(self, index):
        with Image.open(self.img_path_list[index]) as pil_img:
            if (self.train_or_test == 'Train'):
                img = self.transformer['Gray'](self.transformer['Train'](pil_img))
            elif (self.train_or_test == 'Test'):
                img = self.transformer['Gray'](pil_img)
        with Image.open(self.fluid_img_list[index]) as f_img:
            if (self.train_or_test == 'Train'):
                fluid_img = self.transformer['Gray'](self.transformer['Train'](f_img))
            elif (self.train_or_test == 'Test'):
                fluid_img = self.transformer['Gray'](f_img)
        img_ID = self.img_ID_list[index]
        with Image.open(self.seg_img_path_list[index]) as seg_pil_img:
            if (self.train_or_test == 'Train'):
                seg_label = self.transformer['SegTrain'](seg_pil_img)
            elif (self.train_or_test == 'Test'):
                seg_label = self.transformer['SegTest'](seg_pil_img)
            seg_label = np.array(seg_label)
            seg_label[seg_label > 0] = 1
            seg_label[seg_label == 0] = 0
            seg_label = torch.unsqueeze(torch.LongTensor(seg_label), 0)





        return img_ID, img, fluid_img, seg_label

    def __len__(self):
        return len(self.img_path_list)





def moduleTest():
    data_root = "../ResearchData/UltraImageUSFullTest/UltraImageCropWithoutMannualResize"
    seg_root = "../BlurBinaryLabel/"
    us_path = '../ResearchData/data_ultrasound_1.csv'
    fluid_root = "../flowImage/"
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    rf_sort_list = ['SizeOfPlaqueLong', 'SizeOfPlaqueShort', 'DegreeOfCASWtihDiameter', 'Age', 'PSVOfCCA', 'PSVOfICA',
                    'DiameterOfCCA', 'DiameterOfICA', 'EDVOfICA', 'EDVOfCCA', 'RIOfCCA', 'RIOfICA', 'IMT', 'IMTOfICA',
                    'IMTOfCCA', 'Positio0fPlaque', 'Sex', 'IfAnabrosis', 'X0Or0']
    train_dataset = SegDataset(
        str(data_root) + 'TEST/', seg_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Test',
    screener=rf_sort_list,
    screen_num=10)
    print(len(train_dataset))
    test_dataset = FluidSegDatasetPureSeg(
        str(data_root) + 'TEST/', seg_root, fluid_root, train_or_test='Test')
    # test_dataset = FluidSegDatasetPureSeg(
    #     str(data_root) + 'TEST/', seg_root, fluid_root, us_path=us_path, num_classes=NUM_CLASSES, train_or_test='Test',
    #     screener=rf_sort_list, screen_num=10)

    train_dataloader = data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # target为一个batch的值
    target = iter(train_dataloader).next()
    DEVICE = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
    img = target[1]
    seg_label = target[2]
    print(img.shape)
    print(seg_label.shape)
    print(seg_label[0])
    # imshow(img)
    imshow(seg_label[0])

def imshow(tensor, title=None):
    # plt.ion()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    # image = image.float()
    image = transforms.ToPILImage()(image.float()).convert('L')
    # transforms.ToPILImage()(output[j][0]).convert('L')
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(100) 

if __name__ == '__main__':
    moduleTest()
