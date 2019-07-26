from fastai.tabular import *
from fastai.vision import *
from sklearn.model_selection import StratifiedKFold
from fastai.callbacks import *

from custom_nets import *


def train_cnn(train_idxs, val_idxs, train_val_df, image_paths, feature_config_label, feature_config, network_architecture, save_file_suffix, do_callbacks = True ):

    data_fold = (ImageList.from_df(train_val_df, image_paths[feature_config_label][0], cols=f'path_{0:02d}')
                                .split_by_idxs(train_idxs, val_idxs)
                                .label_from_df(cols='class_label')
                                .transform(get_transforms())
                                .databunch(bs=32) 
                            ).normalize(imagenet_stats)
    
    if(do_callbacks):
        learner_callbacks = [partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.0001, patience=2)]
        training_callbacks = [SaveModelCallback(learn, every='improvement', monitor='accuracy')]
    else:
        learner_callbacks = []
        training_callbacks = []
        
    print("classes:",data_fold.classes, "\nclass count:",data_fold.c, "\ntraining set size:",len(data_fold.train_ds),"\nvalidation set size:", len(data_fold.valid_ds))
    learn = cnn_learner(data_fold, network_architecture, metrics=[error_rate, accuracy], callback_fns = learner_callbacks);


    learn.lr_find()
    learn.recorder.plot()

    learn.fit_one_cycle(10, 1e-3, callbacks = training_callbacks)    # every='epoch' to save after each epoch 
    learn.save(f'stage-1'+save_file_suffix)

    learn.unfreeze()

    learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-5/3), callbacks = training_callbacks)
    learn.save(f'stage-2a'+save_file_suffix)

    learn.load(f'stage-1'+save_file_suffix)
    learn.fit_one_cycle(2, max_lr=slice(1e-4,1e-4/3), callbacks = training_callbacks)
    learn.save(f'stage-2b'+save_file_suffix)

    learn.fit_one_cycle(5, max_lr=slice(1e-5,1e-5/3), callbacks = training_callbacks)
    learn.save(f'stage-3'+save_file_suffix)

    return learn
    

def train_hybrid(train_idxs, val_idxs, train_val_df, image_paths, feature_config_label, feature_config, network_architecture, do_callbacks = True ):

    max_allowed_backbones = 1

    imgLists = []
    for i in range(max_allowed_backbones):
        imgLists.append(ImageList.from_df(train_val_df, path = image_paths[feature_config_label][i], cols=f'path_{i:02d}'))

    tabList = TabularList.from_df(train_val_df, cat_names=[], cont_names = feature_config['num_data'], procs=[Normalize])

    mixed = MixedItemList([imgLists[0], tabList], image_paths[feature_config_label][0], inner_df=imgLists[0].inner_df)\
                        .split_by_idxs(train_idxs, val_idxs)\
                        .label_from_df(cols='class_label')
    #.transform([[get_transforms()[0], []], [get_transforms()[1], []]], size=224)

    data_fold = mixed.databunch(no_check=True, bs=32, num_workers=0) # num_workers=0 here just to get errors more quickly
    norm, denorm = normalize_custom_funcs(*imagenet_stats)
    data_fold.add_tfm(norm) # normalize images

    print("classes:",data_fold.classes, "\nclass count:",data_fold.c, "\ntraining set size:",len(data_fold.train_ds),"\nvalidation set size:", len(data_fold.valid_ds))

    # it = iter(data.valid_dl)
    # for item, label in it:
    #     print(label)        

    model = network_architecture#ImageTabularTextModel(3)
    learn = ImageTabularLearner(data_fold, model, metrics=[error_rate])

    learn.lr_find()
    learn.recorder.plot()
    learn.fit_one_cycle(10, .4e-3)
    learn.save('resnet34_stage_1_im_sk_ypr')

    # learn.fit_one_cycle(5, .7e-3)
    learn.load('resnet34_stage_1_im_sk_ypr');
    learn.unfreeze()
    learn.lr_find()
    learn.recorder.plot()
    learn.fit_one_cycle(10, max_lr=slice(1.1e-6,1.1e-6/3))
    learn
    
    return learn