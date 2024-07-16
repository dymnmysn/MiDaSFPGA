from midasfpga.configs.midas_config import Midas_Config
from midasfpga.utils.func_utils import get_midas_transform, get_model, infer_sample
from midasfpga.utils.fpga_utils import quantize
from PIL import Image

if __name__=='__main__':
    conf = Midas_Config()
    model, device = get_model(conf)
    transform = get_midas_transform()
    #infer_sample(model, transform) #Uncomment to save sample inference results

    sample_img = transform(Image.open(conf.SAMPLE_IMAGE_PATH)).unsqueeze(0).to(device)
    quantize(conf.TARGET, conf.QUANT_MODE_CALIB, conf.QUANT_DIR, model, sample_img, device, deploy_check=True)
    quantize(conf.TARGET, conf.QUANT_MODE_TEST, conf.QUANT_DIR, model, sample_img, device, deploy_check=False)
