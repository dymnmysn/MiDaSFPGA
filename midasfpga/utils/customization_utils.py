import torch.nn as nn

def replace_relu6_with_hardtanh(model):
    def replace_recursive(module):
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.ReLU6):
                setattr(module, name, nn.Hardtanh(min_val=0, max_val=6))
            else:
                replace_recursive(child_module)

    replace_recursive(model)
    return model

def replace_dw_layers(model):
    layers = [  model.pretrained.layer1[0],
                model.pretrained.layer1[4][0].conv_dw,
                model.pretrained.layer2[0][0].conv_dw,
                model.pretrained.layer3[0][0].conv_dw,
                model.pretrained.layer4[0][0].conv_dw
            ]

    newlayers = []

    for custom_conv in layers:
        p = custom_conv.pad.padding
        #p = 1 if min(p[:2]) == 0 else 2
        p = (0,1,0,1) if min(p[:2]) == 0 else (1,2,1,2)
        new_conv_layer = nn.Sequential(
                            nn.ZeroPad2d(p),
                            nn.Conv2d( in_channels=custom_conv.in_channels, 
                                    out_channels=custom_conv.out_channels, 
                                    kernel_size=custom_conv.kernel_size,
                                    stride=custom_conv.stride,
                                    #padding = p,
                                    bias = custom_conv.bias,
                                    dilation = custom_conv.dilation,
                                    groups = custom_conv.groups)
        )
                                
        new_conv_layer[1].weight.data.copy_(custom_conv.weight.data)
                                
        newlayers.append(new_conv_layer)

    model.pretrained.layer1[0] = newlayers[0]
    model.pretrained.layer1[4][0].conv_dw = newlayers[1]
    model.pretrained.layer2[0][0].conv_dw = newlayers[2]
    model.pretrained.layer3[0][0].conv_dw = newlayers[3]
    model.pretrained.layer4[0][0].conv_dw = newlayers[4]
    
    return model

def forward_hook(module, input, output):
    print(f'{module.__class__.__name__} output shape:', output.shape)

def print_hook(model):
    for _, layer in model.named_modules():
        layer.register_forward_hook(forward_hook)

