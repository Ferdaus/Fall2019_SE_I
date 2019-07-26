from fastai.vision import *
from sklearn.model_selection import StratifiedKFold
from fastai.callbacks import *
from fastai.tabular import *

def _normalize_images_batch(b:Tuple[Tensor,Tensor], mean:FloatTensor, std:FloatTensor)->Tuple[Tensor,Tensor]:
    "`b` = `x`,`y` - normalize `x` array of imgs and `do_y` optionally `y`."
    x,y = b
    mean,std = mean.to(x[0].device),std.to(x[0].device)
    x[0] = normalize(x[0],mean,std)
    return x,y

def normalize_custom_funcs(mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)->Tuple[Callable,Callable]:
    "Create normalize/denormalize func using `mean` and `std`, can specify `do_y` and `device`."
    mean,std = tensor(mean),tensor(std)
    return (partial(_normalize_images_batch, mean=mean, std=std),
            partial(denormalize, mean=mean, std=std))


class ImageTabularModel_YPR(nn.Module):
    def __init__(self, n_cont, backbone):
        super().__init__()
        
        
        layers = {3:[512, 256], 139:[512, 512, 256]} 
        
        self.cnn = create_body(backbone[0])
        
        nf = num_features_model(self.cnn) * 2
        drop = .5

        self.tab = TabularModel({}, n_cont, 128, layers[n_cont])

        self.reduce = nn.Sequential(*([AdaptiveConcatPool2d(), Flatten()] + bn_drop_lin(nf, 512, bn=True, p=drop, actn=nn.ReLU(inplace=True))))
        self.merge = nn.Sequential(*bn_drop_lin(512 + 128, 128, bn=True, p=drop, actn=nn.ReLU(inplace=True)))
        self.final = nn.Sequential(*bn_drop_lin(128, 2, bn=False, p=0., actn=None))
        #self.final = nn.Sequential(*bn_drop_lin(512, 2, bn=False, p=0., actn=None))
        #print(self)
        
    def forward(self, img:Tensor, x:Tensor) -> Tensor:
        #print(img.shape)        
        imgCnn = self.cnn(img)
        #print(imgCnn.shape)
        imgLatent = self.reduce(imgCnn)
        #print(imgLatent.shape)
        tabLatent = self.tab(x[0], x[1])
        #print(tabLatent.shape)
        
        cat = torch.cat([imgLatent, F.relu(tabLatent)], dim=1)
        #print(cat.shape)
        
        pred = self.final(self.merge(cat))
        #pred = torch.sigmoid(pred)  # making sure this is in the range 0-4
        #pred = torch.sigmoid(self.final(self.reduce(imgCnn)))
        #print(pred)
        return pred #torch.softmax(torxh.zeros(1,2))
        
    def reset(self):
        for c in self.children():
            if hasattr(c, 'reset'): c.reset()
                
                
