import torch
from tqdm import tqdm
            
def get_line_errors(out_full, out_patch, full_ids, patch_ids, threshold=1e-6):

    errors = []
    for (line_f, line_p) in zip([*range(full_ids[0],full_ids[1])], [*range(patch_ids[0],patch_ids[1])]):
        out_p_line = out_patch[0,:,line_p,:]
        out_line = out_full[0,:,line_f,:]
        errors.append(torch.abs(out_line - out_p_line).max())
    return errors        
  
def learn_slice_flip(forward_fn, num_slices=4, slice_id=3, threshold=1e-6, input_shape=(1, 3, 160, 160), debug=False):
    
    input_tensor = torch.rand(input_shape)  
    output_tensor = forward_fn(input_tensor)
    out_h = output_tensor.shape[2]
    input_h = input_tensor.shape[2]
    in_stride, out_stride = max(1, input_h//out_h), max(1, out_h//input_h)
    if debug:
        print(f"Input size: {input_h}, Output size: {out_h} -> Input stride: {in_stride}, output stride: {out_stride}")
    
    if out_h % num_slices != 0:
        raise ValueError("The number of slices must be a divisor of the height of the output tensor")
    
    offset = out_h//num_slices * slice_id
    target_lines = [0, out_h//num_slices]
    
    start_row = input_h//num_slices * slice_id
    end_row =  input_h//num_slices * (slice_id+1)
    
    while True:
        input_tensor_p = input_tensor[:,:,start_row:end_row,:]
        try:
            output_tensor_p = forward_fn(input_tensor_p)
        except:
            # breakpoint()
            raise ValueError("Error in forward function")
        
        errors = get_line_errors(output_tensor, output_tensor_p, [offset, offset + out_h//num_slices], target_lines, threshold=threshold)
        e_first, e_last = errors[0], errors[-1]
        
        if e_last > threshold:
            end_row += in_stride
            if end_row > input_tensor.shape[2]:
                end_row = input_tensor.shape[2]
                print(f"WARNING: End slice is too large - ERROR: {e_last:.4f}")
                e_last = -1
        
        if e_first > threshold:
            start_row -= in_stride
            target_lines[0] += out_stride
            target_lines[1] += out_stride
            if start_row < 0:
                start_row = 0
                print(f"WARNING: Start slice is too small - ERROR: {e_first:.4f}")
                e_first = -1
            
        if e_first <= threshold and e_last <= threshold:
            ram_saving = (1 - (end_row - start_row) / input_h)
            compute_overhead = (end_row - start_row)/(input_h//num_slices)
            return (start_row, end_row), target_lines, ram_saving, compute_overhead
            
def learn_slices(forward_fn, input_shape, num_slices=4, threshold=1e-6):
    slices = []
    ram_saving = 1
    compute_overhead = 0

    for slice_id in tqdm(range(num_slices)):
        input_lines, output_lines, rs, co = learn_slice_flip(forward_fn, num_slices=num_slices, slice_id=slice_id, threshold=threshold, input_shape=input_shape)
        slices.append((input_lines, output_lines))
        ram_saving = min(ram_saving, rs)
        compute_overhead+=co
    compute_overhead /= num_slices
    return slices, ram_saving, compute_overhead

def sliced_forward(forward_fn, slices, example_input, alpha=0):
    # Local Exact
    slices_output = []
    for slice in slices:
        input_lines, output_lines = slice
        input_tensor = example_input[:,:,input_lines[0]:input_lines[1],:]
        try:
            output_tensor = forward_fn(input_tensor)[:,:,output_lines[0]:output_lines[1],:]
        except:
            continue
        slices_output.append(output_tensor)
    full_output_from_slices = torch.cat(slices_output, dim=2)

    # Global Approximate
    if alpha > 0:
        factor = int(len(slices) ** 0.5)
        resized_input = torch.nn.functional.interpolate(example_input, scale_factor=1/factor, mode='bilinear', align_corners=False)
        resized_output = forward_fn(resized_input)
        resized_output = torch.nn.functional.interpolate(resized_output, size=full_output_from_slices.shape[2:], mode='bilinear', align_corners=False)
        full_output_from_slices = alpha * resized_output + (1 - alpha) * full_output_from_slices
    
    return full_output_from_slices