from pytorch_nndct.apis import torch_quantizer

def quantize(target, quant_mode, quant_dir, model, img, device, deploy_check):
    quantizer = torch_quantizer(quant_mode, model, (img), output_dir = quant_dir, device=device, 
                                target=target)
    qmodel = quantizer.quant_model
    _ = qmodel(img)
    quantizer.export_quant_config()
    quantizer.export_xmodel(quant_dir, deploy_check=deploy_check)
    quantizer.export_torch_script(output_dir = quant_dir)
