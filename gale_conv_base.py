import torch
import gale_core

class SlicingModule:
    def __init__(self, model, num_slices=4, threshold=1e-6, split_blocks=1, use_square=False, alpha=0):
        self.model = model
        self.num_slices = num_slices
        self.threshold = threshold
        self.split_blocks = split_blocks
        self.slices_levels = []
        self.learned = False
        self.use_square = use_square
        self.alpha=alpha
        self.subdivision_fn = gale_core.learn_slices
        self.forward_fn = gale_core.sliced_forward
        print(f"Using slicing with {num_slices} slices")
    
    def learn_slices(self, x):
        print("Learning slices")
        self.model.cpu()
        self.slices_levels = []
        level_input = x
        level_slice_num = self.num_slices
        for i in range(self.split_blocks):
            level_in_shape = (1, *level_input.shape[1:])
            try:
                slices, ram_savings, compute_overhead = self.subdivision_fn(self.blocks[i], level_in_shape, level_slice_num, self.threshold)
            except:
                raise ValueError("Error in forward function")
            self.slices_levels.append(slices)
            level_input = self.blocks[i](level_input[:1].cpu())
            if not self.use_square:
                level_slice_num //= 2
            print(f"Block {i} RAM saving: {(1-(1-ram_savings)/(i+1))*100:.2f}%, Compute overhead: {(compute_overhead-1)*100:.2f}%")
        self.model.cuda()
        self.learned = True
    
    def sliced_forward(self, x, debug=False):
        if not self.learned:
            self.learn_slices(x)
        
        for i in range(self.split_blocks):
            y_from_slices = self.forward_fn(self.blocks[i], self.slices_levels[i], x, self.alpha)
            if debug:
                y_full = self.blocks[i](x)
                print(f"==================== Level {i} ====================")
                print(f"Output: {y_full.shape} Full output from slices: {y_from_slices.shape}")
                print(f"Max Error: {torch.abs(y_from_slices - y_full).max()}")
                print(f"MSE: {torch.nn.functional.mse_loss(y_from_slices, y_full)}")
                if y_full.shape != y_from_slices.shape:
                    raise ValueError("Output shapes do not match")
                    # breakpoint()
            
            # breakpoint()
            x = y_from_slices
        
        for i in range(self.split_blocks, len(self.blocks)):
            x = self.blocks[i](x)
        return x
    
    def forward(self, x, debug=False):
        return self.sliced_forward(x, debug)
    
    @property
    def blocks(self):
        """ Define model-specific block sequences """
        raise NotImplementedError

# Define subclasses for specific models
class ResNetSlicing(SlicingModule):
    @property
    def blocks(self):
        return [
            lambda x: self.model.layer1(self.model.maxpool(self.model.act1(self.model.bn1(self.model.conv1(x))))),
            self.model.layer2,
            self.model.layer3,
            self.model.layer4
        ]

class MBNet3Slicing(SlicingModule):
    @property
    def blocks(self):
        return [
            lambda x: self.model.blocks[0](self.model.bn1(self.model.conv_stem(x))),
            *self.model.blocks[1:],
        ]

class EfficientNetSlicing(SlicingModule):
    @property
    def blocks(self):
        return [
            lambda x: self.model.blocks[0](self.model.bn1(self.model.conv_stem(x))),
            *self.model.blocks[1:],
            lambda x: self.model.bn2(self.model.conv_head(x))
        ]

class FastViTSlicing(SlicingModule):
    @property
    def blocks(self):
        return [
            lambda x: self.model.stem(x),
            *self.model.stages,
            self.model.final_conv
        ]

class MITEfficientViTSlicing(SlicingModule):
    @property
    def blocks(self):
        return [
            lambda x: self.model.stem(x),
            *self.model.stages
        ]
