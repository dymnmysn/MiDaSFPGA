class Midas_Config:
    #These are for both PC and FPGA parts
    IMG_W = 256
    IMG_H = 192
    BATCH_SIZE = 32
    BACKBONE = 'resnet18'
    TEST_BATCH_SIZE = 1000
    SEED = 1
    LOG_INTERVAL = 10
    RESUME = ''
    DATA_ROOT = 'data'
    QUANT_DIR = 'quantization_results/build/quantized'
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
    MODEL_WEIGHTS = '/kaggle/working/MiDaSFPGA/weights/midas_v21_small_256.pt'
    SAMPLE_IMAGE_PATH = 'data/rgb_00065.png'
    TARGET = FP_4096

    #These are for FPGA board
    FPGA_XMODEL_PATH = '/home/ubuntu/Downloads/target_kv260/fcn8/model/fcn8.xmodel'
    FPGA_DATA_DIR = '/home/ubuntu/Downloads/data'
    FPGA_INDEX_LENGTH = 6
    FPGA_PARTITION_LENGTH = 109

    
