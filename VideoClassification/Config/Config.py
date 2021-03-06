WHERE = 'SERVER'

if WHERE == 'LOCAL':
    ''' local pc '''
    vgg16pretrainfile = '/home/itrc/BackUp/pretrained/vgg16-397923af.pth'
    resnet101_pretrainfile = '/home/itrc/BackUp/pretrained/resnet101-5d3b4d8f.pth'
    resnet152_pretrainfile = '/home/itrc/BackUp/pretrained/resnet152-b121ed2d.pth'

    ExWorkSpace = '/home/itrc/ExperimentsWorkspace/VC/'
    LOGSpace = ExWorkSpace

    Code_root = '/home/itrc/Desktop/Development/VideoClassification/'
    UCF101_images_root = './'
    EX_ID = '/EX_test1'

    ExWorkSpace = '/home/itrc/ExperimentsWorkspace/VC/'
    LOGSpace = ExWorkSpace

elif WHERE == 'SERVER':

    ''' server '''
    SERVER_ID = '202'
    Server_Root_Path = '/ssd/2/UCF_Data/'
    # Server_Root_Path = '/datacenter/1/UCF_Data/'
    if SERVER_ID == '211':
        Server_Root_Path = '/ssd/1/UCF_Data/'
    elif SERVER_ID == '210' or SERVER_ID == '209':
        Server_Root_Path = '/ssd/2/UCF_Data/'
    elif SERVER_ID == '202':
        Server_Root_Path = '/datacenter/1/UCF_Data/'

    vgg16pretrainfile = Server_Root_Path + 'pretrainedmodel/vgg16-397923af.pth'
    resnet101_pretrainfile = Server_Root_Path + 'pretrainedmodel/resnet101-5d3b4d8f.pth'
    resnet152_pretrainfile = Server_Root_Path + 'pretrainedmodel/resnet152-b121ed2d.pth'
    densenet169_pretrainfile = Server_Root_Path + 'pretrainedmodel/densenet169-6f0f7f60.pth'
    densenet201_pretrainfile = Server_Root_Path + 'pretrainedmodel/densenet201-4c113574.pth'

    ExWorkSpace = Server_Root_Path + 'EX/'
    LOGSpace = Server_Root_Path + 'LOG/'
    Code_root = Server_Root_Path + 'Code/VideoClassification/'

    UCF101_images_root = Server_Root_Path + 'UCF101_images/'

    # EX_ID = '/VGG16_Temporal_EX3'
    # EX_ID = '/ResNet101_Spatial_EX2'
    # EX_ID = '/ResNet101_Temporal_EX2'
    # EX_ID = '/ResNet152_Temporal_WITHOUT_FC_EX3'
    # EX_ID = '/ResNet152_Spatial_WITHOUT_FC_EX2'
    # EX_ID = '/C3D_Without_FC'
    # EX_ID = '/DenseNet169_Temporal_WITHOUT_FC_EX1'
    EX_ID = '/DPN131_EX1'

    # LOAD_SAVED_MODE_PATH = '/datacenter/1/UCF_Data/EX/ResNet101_Spatial_EX1/ResNet101_Spatial_01.pt'
    # LOAD_SAVED_MODE_PATH = '/ssd/2/UCF_Data/EX/ResNet152_Temporal_WITHOUT_FC_EX2/ResNet152_Temporal_EX1_18.pt'
    LOAD_SAVED_MODE_PATH = None
    LOAD_SAVED_MODE_PATH_2 = None

    MySQL_Config = dict(host='10.1.1.202', user='root', password='123456', db='ucf101', charset='utf8', autocommit=True)

elif WHERE == 'DOCKER':
    raise NotImplementedError
else:
    raise NotImplementedError
