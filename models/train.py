from models.unet_wsl.train_model import model_execution as train_unet
from models.unet_VGG16.train_model import model_execution as train_unet_vgg16
from models.unet_ResNet50.train_model import model_execution as train_unet_resnet50
from models.unet_plus_plus.train_model import model_execution as train_unet_plus_plus
from models.unet_MobileNetV2.train_model import model_execution as train_unet_mobilenetv2
from models.unet_ffc.train_model import model_execution as train_unet_ffc
from models.segnet_VGG16.train_model import model_execution as train_segnet_vgg16
from models.segnet.train_model import model_execution as train_segnet
from models.res_unet_plus_plus.train_model import model_execution as train_res_unet_plus_plus
from models.deeplabv3_plus.train_model import model_execution as train_deeplabv3_plus

from unet_wsl.train_model import load_saved_model as load_unet_model
from unet_ffc.train_model import load_saved_model as load_unet_ffc_model
from unet_VGG16.train_model import load_saved_model as load_unet_VGG16_model
from unet_ResNet50.train_model import load_saved_model as load_unet_ResNet50_model
from unet_MobileNetV2.train_model import load_saved_model as load_unet_MobileNetV2_model
from unet_plus_plus.train_model import load_saved_model as load_unet_plus_plus_model
from segnet.train_model import load_saved_model as load_segnet_model
from segnet_VGG16.train_model import load_saved_model as load_segnet_VGG16_model
from res_unet_plus_plus.train_model import load_saved_model as load_res_unet_plus_plus_model
from deeplabv3_plus.train_model import load_saved_model as load_deeplabv3_plus_model

def load_saved_unet_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_wsl\config.yaml'
    return load_unet_model(config_file)

def load_saved_unet_VGG16_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_VGG16\config.yaml'
    return load_unet_VGG16_model(config_file)

def load_saved_unet_ResNet50_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_ResNet50\config.yaml'
    return load_unet_ResNet50_model(config_file)

def load_saved_unet_plus_plus_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_plus_plus\config.yaml'
    return load_unet_plus_plus_model(config_file)

def load_saved_unet_MobileNetV2_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_MobileNetV2\config.yaml'
    return load_unet_MobileNetV2_model(config_file)

def load_saved_unet_ffc_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_ffc\config.yaml'
    return load_unet_ffc_model(config_file)

def load_saved_segnet_VGG16_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet_VGG16\config.yaml'
    return load_segnet_model(config_file)

def load_saved_segnet_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet\config.yaml'
    return load_segnet_VGG16_model(config_file)

def load_saved_res_unet_plus_plus_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\res_unet_plus_plus\config.yaml'
    return load_res_unet_plus_plus_model(config_file)

def load_saved_deeplabv3_plus_model():
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\deeplabv3_plus\config.yaml'
    return load_deeplabv3_plus_model(config_file)

def train_all_models():
    print('UNET model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_wsl\config.yaml'
    train_unet(config_file)
    print('UNET-VGG16 model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_VGG16\config.yaml'
    train_unet_vgg16(config_file)
    print('UNET-ResNet50 model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_ResNet50\config.yaml'
    train_unet_resnet50(config_file)
    print('UNET++ model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_plus_plus\config.yaml'
    train_unet_plus_plus(config_file)
    print('UNET-MobileNetV2 model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_MobileNetV2\config.yaml'
    train_unet_mobilenetv2(config_file)
    print('UNET-FCC model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_ffc\config.yaml'
    train_unet_ffc(config_file)
    print('SegNet-VGG16 model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet_VGG16\config.yaml'
    train_segnet_vgg16(config_file)
    print('SegNet model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet\config.yaml'
    train_segnet(config_file)
    print('ResUNET++ model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\res_unet_plus_plus\config.yaml'
    train_res_unet_plus_plus(config_file)
    print('DeepLabV3+ model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\deeplabv3_plus\config.yaml'
    train_deeplabv3_plus(config_file)
    print('All model training completed')

if __name__ == "__main__":
    print('UNET model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_wsl\config.yaml'
    train_unet(config_file)
    print('UNET-VGG16 model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_VGG16\config.yaml'
    train_unet_vgg16(config_file)
    print('UNET-ResNet50 model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_ResNet50\config.yaml'
    train_unet_resnet50(config_file)
    print('UNET++ model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_plus_plus\config.yaml'
    train_unet_plus_plus(config_file)
    print('UNET-MobileNetV2 model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_MobileNetV2\config.yaml'
    train_unet_mobilenetv2(config_file)
    print('UNET-FCC model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\unet_ffc\config.yaml'
    train_unet_ffc(config_file)
    print('SegNet-VGG16 model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet_VGG16\config.yaml'
    train_segnet_vgg16(config_file)
    print('SegNet model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\segnet\config.yaml'
    train_segnet(config_file)
    print('ResUNET++ model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\res_unet_plus_plus\config.yaml'
    train_res_unet_plus_plus(config_file)
    print('DeepLabV3+ model training...')
    config_file = 'C:\\Users\AdikariAdikari\PycharmProjects\Segmentation\models\\deeplabv3_plus\config.yaml'
    train_deeplabv3_plus(config_file)
    print('All model training completed')

