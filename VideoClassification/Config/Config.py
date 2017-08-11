
WHERE = 'SERVER'

if WHERE=='LOCAL':
    ''' local pc '''
    vgg16pretrainfile = '/home/itrc/BackUp/pretrained/vgg16-397923af.pth'
    resnet101_pretrainfile = '/home/itrc/BackUp/pretrained/resnet101-5d3b4d8f.pth'
    resnet152_pretrainfile = '/home/itrc/BackUp/pretrained/resnet152-b121ed2d.pth'

    ExWorkSpace = '/home/itrc/ExperimentsWorkspace/VC/'
    LOGSpace = ExWorkSpace

    Code_root = '/home/itrc/Desktop/Development/VideoClassification/'
    UCF101_images_root = './'
    EX_ID='/EX_test1'

    ExWorkSpace = '/home/itrc/ExperimentsWorkspace/VC/'
    LOGSpace = ExWorkSpace

elif WHERE=='SERVER':

    ''' server '''
    # Server_Root_Path = '/ssd/2/UCF_Data/'
    Server_Root_Path = '/datacenter/1/UCF_Data/'

    vgg16pretrainfile = Server_Root_Path+'pretrainedmodel/vgg16-397923af.pth'

    ExWorkSpace = Server_Root_Path+'EX/'

    LOGSpace = Server_Root_Path+'LOG/'

    Code_root = Server_Root_Path+'Code/VideoClassification/'

    UCF101_images_root = Server_Root_Path+'UCF101_images/'

    # EX_ID = '/ResNet101_Spatial_EX2'
    # EX_ID = '/ResNet101_Temporal_EX2'
    # EX_ID = '/C3D_EX8'
    EX_ID = '/DenseNet201_S_EX2'

    # LOAD_SAVED_MODE_PATH = '/datacenter/1/UCF_Data/EX/ResNet101_Spatial_EX1/ResNet101_Spatial_01.pt'
    LOAD_SAVED_MODE_PATH = None
    LOAD_SAVED_MODE_PATH_2 = None

    MySQL_Config = dict( host='10.1.1.202', user='root', password='123456', db='ucf101', charset='utf8', autocommit=True )

elif WHERE=='DOCKER':
    raise NotImplementedError
else:
    raise NotImplementedError
