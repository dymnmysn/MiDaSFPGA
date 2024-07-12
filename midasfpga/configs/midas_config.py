
class Midas_Config:
    IMG_W = 256
    IMG_H = 192
    BATCH_SIZE = 32
    BACKBONE = 'resnet18'
    TEST_BATCH_SIZE = 1000
    SEED = 1
    LOG_INTERVAL = 10
    RESUME = ''
    DATA_ROOT = '/workspace/demodel/derlitoplu/build/data'
    QUANT_DIR = '/workspace/demodel/derlitoplu/build/quantized'
    QUANT_MODE = 'float'
    QUANT_MODE_CALIB = 'calib'
    QUANT_MODE_TEST = 'test'
    DEVICE = 'gpu' 
    DEPLOY = False
    CUDA_HOME = '/usr/local/cuda'
    TARGET_4096 = "DPUCZDX8G_ISA1_B4096" 
    FP_4096 = "0x101000016010407"
    TARGET_3136 = "DPUCZDX8G_ISA1_B3136" 
    FP_3136 = "0x101000016010406"
    MODEL_WEIGHTS = '/workspace/demodel/weights/midas_v21_small_256.pt'
    SAMPLE_IMAGE_PATH = '/workspace/demodel/dog.jpg'

    TARGET = FP_4096

