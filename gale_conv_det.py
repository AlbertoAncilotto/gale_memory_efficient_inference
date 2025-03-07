import torch
import gale_core


class SlicedModel:
    def __init__(self, model, num_slices=16, threshold=1, split_blocks=4, alpha=0):
        self.model = model
        self.num_slices = num_slices
        self.threshold = threshold
        self.split_blocks = split_blocks
        self.alpha = alpha
        self.slices = []
        self.learned = False
        model.model.predict = self.forward

    def learn_slices(self, x):
        print("Learning slices")
        x = x.cpu()
        self.model.cpu()
        num_slices = self.num_slices
        for id_m, m in enumerate(self.model.model.model[:self.split_blocks]): #thanks, ultralytics
            if m.f != -1:
                raise ValueError("Unexpected skip connection input in sliced section")
            y_full = m(x)
            slices, ram_savings, compute_overhead = gale_core.learn_slices(m, (1, *x.shape[1:]), num_slices, self.threshold)
            self.slices.append(slices)
            y_from_slices = gale_core.sliced_forward(m, self.slices[id_m], x, alpha=self.alpha)
            print(f"==================== Level {id_m} ====================")
            if num_slices == 1:
                print(f"WARNING: level {id_m} has only one slice")
            print(f"RAM saving: {(1-(1-ram_savings)/(id_m+1))*100:.2f}%, Compute overhead: {(compute_overhead-1)*100:.2f}%, MSE: {torch.nn.functional.mse_loss(y_full, y_from_slices)}")
            num_slices = max(1, num_slices//2)
            x = y_from_slices
            
            #we do not manage the skip connections in the sliced section
        self.learned = True
        self.model.cuda()
        x = x.cuda()

    def forward_features_split(self, x, y):
        for id_m, m in enumerate(self.model.model.model[:self.split_blocks]):
            x = gale_core.sliced_forward(m, self.slices[id_m], x, alpha=self.alpha)
            y.append(x if m.i in self.model.model.save else None)
        return x

    def forward_features_B(self, x, y):
        for m in self.model.model.model[self.split_blocks:]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.model.model.save else None)
        return x

    def forward(self, x, *args, **kwargs):
        y=[]
        if not self.learned:
            self.learn_slices(x)
        return self.forward_features_B(self.forward_features_split(x, y), y)
    
    def __call__(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)



class YoloSliced(SlicedModel):
    pass 


class RTDetrSliced(SlicedModel):
    def forward_features_B(self, x, y, batch=None):
        for m in self.model.model.model[self.split_blocks:-1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.model.model.save else None)
        head = self.model.model.model[-1]
        return head([y[j] for j in head.f], batch)

if __name__ == '__main__':
    from ultralytics import YOLO, RTDETR
    import cv2

    model = YOLO('yolo11n.pt')
    sliced_model = YoloSliced(model, alpha=0.1)

    # model = RTDETR("rtdetr-l.pt")
    # sliced_model = RTDetrSliced(model)

    image_path = 'test_img.jpg'
    image = cv2.imread(image_path)

    results = model(image)[0]

    sliced_results = sliced_model(image)[0]

