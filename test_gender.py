#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on NPM3D dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os

# Dataset
from datasets.GenderData import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.trainer import ModelTrainer
from utils.tester import ModelTester
from utils.metrics import fast_confusion, IoU_from_confusions
from models.architectures import KPFCNN
from sklearn.cluster import DBSCAN
from models.DGCNN_classifier_backbone import DMS_DGCNN
import glob

# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#

class NPM3DConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'NPM3D'

    # Number of classes in the dataset (This value is overwritten by dataset class when Initializating dataset).
    num_classes = None

    # Type of task performed on this dataset (also overwritten)
    dataset_task = ''

    # Number of CPU threads for the input pipeline
    input_threads = 10

    #########################
    # Architecture definition
    #########################

    # # Define layers
    architecture = ['resnetb',
                     'resnetb_strided',
                     'resnetb',
                     'resnetb_strided',
                     'resnetb',
                     'resnetb_strided',
                     'resnetb',
                     'resnetb_strided',
                     'resnetb',
                     'nearest_upsample',
                     'unary',
                     'nearest_upsample',
                     'unary',
                     'nearest_upsample',
                     'unary',
                     'nearest_upsample',
                     'unary']

    ###################
    # KPConv parameters
    ###################

    # Number of kernel points
    num_kernel_points = 15

    # Radius of the input sphere (decrease value to reduce memory cost)
    in_radius = 1.5

    # Size of the first subsampling grid in meter (increase value to reduce memory cost)
    first_subsampling_dl = 0.06

    # Radius of convolution in "number grid cell". (2.5 is the standard value)
    conv_radius = 2.5

    # Radius of deformable convolution in "number grid cell". Larger so that deformed kernel can spread out
    deform_radius = 5.0

    # Radius of the area of influence of each kernel point in "number grid cell". (1.0 is the standard value)
    KP_extent = 1.2

    # Behavior of convolutions in ('constant', 'linear', 'gaussian')
    KP_influence = 'linear'

    # Aggregation function of KPConv in ('closest', 'sum')
    aggregation_mode = 'sum'

    # Choice of input features
    first_features_dim = 128
    in_features_dim = 1

    # Can the network learn modulations
    modulated = False

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.02

    # Deformable offset loss
    # 'point2point' fitting geometry by penalizing distance from deform point to input points
    # 'point2plane' fitting geometry by penalizing distance from deform point to input point triplet (not implemented)
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0              # Multiplier for the fitting/repulsive loss
    deform_lr_factor = 0.1                  # Multiplier for learning rate applied to the deformations
    repulse_extent = 1.2                    # Distance of repulsion for deformed kernel points

    #####################
    # Training parameters
    #####################

    # Maximal number of epochs
    max_epoch = 500

    # Learning rate management
    learning_rate = 1e-2
    momentum = 0.98
    lr_decays = {i: 0.1 ** (1 / 150) for i in range(1, max_epoch)}
    grad_clip_norm = 100.0

    # Number of batch (decrease to reduce memory cost, but it should remain > 3 for stability)
    batch_num = 6

    # Number of steps per epochs
#     epoch_steps = 500
    epoch_steps = 500

    # Number of validation examples per epoch
    validation_size = 50

    # Number of epoch between each checkpoint
    checkpoint_gap = 50

    # Augmentations
    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_noise = 0.001
    augment_color = 0.8

    # The way we balance segmentation loss
    #   > 'none': Each point in the whole batch has the same contribution.
    #   > 'class': Each class has the same contribution (points are weighted according to class balance)
    #   > 'batch': Each cloud in the batch has the same contribution (points are weighted according cloud sizes)
    segloss_balance = 'none'

    # Do we nee to save convergence
    saving = True
    saving_path = ''


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Choose here if you want to start training from a previous snapshot (None for new training)
    # previous_training_path = 'Log_2020-03-19_19-53-27'
    previous_training_path = ''

    # Choose index of checkpoint to start from. If None, uses the latest chkp
    chkp_idx = None
    if previous_training_path:

        # Find all snapshot in the chosen training folder
        chkp_path = os.path.join('results', previous_training_path, 'checkpoints')
        chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

        # Find which snapshot to restore
        if chkp_idx is None:
            chosen_chkp = 'current_chkp.tar'
        else:
            chosen_chkp = np.sort(chkps)[chkp_idx]
        chosen_chkp = os.path.join('results', previous_training_path, 'checkpoints', chosen_chkp)

    else:
        chosen_chkp = None

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    # Initialize configuration class
    config = NPM3DConfig()
    if previous_training_path:
        config.load(os.path.join('results', previous_training_path))
        config.saving_path = None

    # Get path from argument if given
    if len(sys.argv) > 1:
        config.saving_path = sys.argv[1]

    # Initialize datasets
    training_dataset = NPM3DDataset(config, set='training', use_potentials=True)
    test_dataset = NPM3DDataset(config, set='validation', use_potentials=True)

    # Initialize samplers
    training_sampler = NPM3DSampler(training_dataset)
    test_sampler = NPM3DSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=NPM3DCollate,
                                 num_workers=config.input_threads,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=NPM3DCollate,
                             num_workers=config.input_threads,
                             pin_memory=True)

    # Calibrate samplers
    training_sampler.calibration(training_loader, verbose=True)
    test_sampler.calibration(test_loader, verbose=True)

    # Optional debug functions
    # debug_timing(training_dataset, training_loader)
    # debug_timing(test_dataset, test_loader)
    # debug_upsampling(training_dataset, training_loader)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, training_dataset.label_values, training_dataset.ignored_labels)

    debug = False
    if debug:
        print('\n*************************************\n')
        print(net)
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    chosen_chkp = './results/Log_2022-11-06_21-08-22/checkpoints/current_chkp.tar'
#     chosen_chkp = None
    
    
    # Define a trainer class
#     trainer = ModelTrainer(net, config, chkp_path=chosen_chkp)
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart training')
    print('**************')

    # Training
#     trainer.train(net, training_loader, test_loader, config)
#     tester.cloud_segmentation_test(net, test_loader, config, num_votes=10)
    
    print("Segmentation testing done...")
    
    
    print("Loading Classifier...")
    model = DMS_DGCNN(NUM_CHANNELS=2)
    model.load_weights('./results/train_checkpoints/DMS_DGCNN_train_v2')
    
    lb_map = {0: 1, 1: 4}
    data_dir = sorted(glob.glob('./test/gender_predictionsV3/*'))
    # data_dir = sorted(glob.glob('../../KPConv-PyTorch/results/Log_2022-10-21_15-41-06/val_preds_250/*'))
    data_dir_orig = sorted(glob.glob('../GenderData/ply_files/*.ply'))[25:31]
    clus_points = []
    clus_labels = []
    for i, file in enumerate(data_dir):
        print(f'Processing file: {file}')
        cloud = read_ply(file)
        cloud_orig = read_ply(data_dir_orig[i])

        if cloud.shape[0] != cloud_orig.shape[0]:
            print('Not same dimension')
            break

        lb = cloud['preds'] 
        x = cloud['x']
        y = cloud['y']
        z = cloud['z']

    #     lb_orig = cloud['class']
        lb_orig = cloud_orig['class']

        arr = np.vstack((x, y, z)).T

        arr_lb = np.hstack((arr, np.expand_dims(lb, axis=-1)))

        indices = np.arange(0, arr_lb.shape[0]).astype(np.int32)

        arr_lb_ind = np.hstack((arr_lb, np.expand_dims(indices, axis=-1)))

        male_arr = arr_lb_ind[lb == 1]
        female_arr  = arr_lb_ind[lb == 4]

        male_data = male_arr[:, :3]
        male_ind = male_arr[:, -1]

        female_data = female_arr[:, :3]
        female_ind = female_arr[:, -1]

        data = np.vstack((male_data, female_data))
        data_ind = np.concatenate((male_ind, female_ind), axis=0).astype(np.int32)

        print(f"Processing clusters...")
        clusterer = DBSCAN(eps=0.5, min_samples=2)
        clusterer.fit_predict(data)

        data_id = np.concatenate((data, np.expand_dims(data_ind, axis=-1)), axis=-1)

        clusters = clusterer.labels_
        n_clusters = np.unique(clusters)

        inst_data = np.empty(shape=(0,5))

        NUM_POINTS = 1000

        print("Processing clusters through classifier...")
        for i in n_clusters:
            instance_pt = data_id[clusters == i]
            # (1,)
            print(instance_pt[:, :3].shape)
            if instance_pt.shape[0] >= NUM_POINTS:
                idx = np.random.choice(instance_pt.shape[0], NUM_POINTS, replace=False)
                label = np.argmax(model(np.expand_dims(instance_pt[idx][:, :3], axis=0), training=False), axis=-1)
                label = lb_map[int(label[0])]
    #             print(label)
                instance_id = instance_pt[:, -1].astype(np.int32)
                np.put(lb, instance_id.astype(np.int32), label)

                label_full = np.full((instance_pt.shape[0], 1), label)

                inst_data = np.vstack((inst_data, np.concatenate((instance_pt, label_full), axis=-1)))

    #             print(np.bincount(lb_orig[instance_id]))
    #             clus_labels.append(lb_map[int(np.bincount(lb_orig[instance_id]).argmax())])
    #             clus_points.append(instance_pt[idx])

        write_ply(f'./complete_{i}.ply', [inst_data], ['x', 'y', 'z', 'indices', 'label'])

        print(f"Writing ply: {file.split('/')[-1]}")
        write_ply(f"./SegClusClass_preds/{file.split('/')[-1]}", [arr, lb, lb_orig], ['x', 'y', 'z', 'preds_new', 'class'])
        
        
        
    
    print("Computing Confusions...")
     # Regroup confusions
    label_to_names = {
                            0: "buildings",
                            1: "female",
                            2: "furniture",
                            3: "ground",
                            4: "male",
                            5: "trees",
                            6: "unclassified1",
                            7: "vehicle",
                            8: "unclassified2"

            }


    ignored_labels = np.array([6, 8])

    label_values = np.sort([k for k, v in label_to_names.items()])

    nc_total = len(label_to_names)

    nc_model = len(label_to_names) - len(ignored_labels)

    # pred_dir = sorted(glob.glob('./KPConv-PyTorch/results/Log_2022-10-21_15-41-06/val_preds_250/*'))
    pred_dir = sorted(glob.glob('./SegClusClass_preds/*'))
    validation_labels = []

    Confs = []
    for file in pred_dir:
        cloud = read_ply(file)

        preds = cloud['preds_new']
        true = cloud['class']

        validation_labels += [true]
        Confs += [fast_confusion(true, preds, label_values)]
        print(preds.shape, true.shape)


    val_proportions = np.zeros(nc_model, dtype=np.float32)
    i = 0
    for label_value in label_values:
        if label_value not in ignored_labels:
            val_proportions[i] = np.sum([np.sum(labels == label_value)
                                        for labels in validation_labels])
            i += 1

    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

    for l_ind, label_value in reversed(list(enumerate(label_values))):
        if label_value in ignored_labels:
            C = np.delete(C, l_ind, axis=0)
            C = np.delete(C, l_ind, axis=1)


    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

    IoUs_seg = IoU_from_confusions(C)
    mIoU_seg = np.mean(IoUs_seg)
    
    
     # Regroup confusions
    label_to_names = {
                            0: "buildings",
                            1: "female",
                            2: "furniture",
                            3: "ground",
                            4: "male",
                            5: "trees",
                            6: "unclassified1",
                            7: "vehicle",
                            8: "unclassified2"

            }


    ignored_labels = np.array([6, 8])

    label_values = np.sort([k for k, v in label_to_names.items()])

    nc_total = len(label_to_names)

    nc_model = len(label_to_names) - len(ignored_labels)

    # pred_dir = sorted(glob.glob('./KPConv-PyTorch/results/Log_2022-10-21_15-41-06/val_preds_250/*'))
    # pred_dir = sorted(glob.glob('./SegClusClass_val_predV2/*'))
    # pred_dir_orig = sorted(glob.glob('./'))

    pred_dir_orig = sorted(glob.glob('./test/gender_predictionsV3/*'))
    # data_dir = sorted(glob.glob('../../KPConv-PyTorch/results/Log_2022-10-21_15-41-06/val_preds_250/*'))
    # pred_dir_orig = sorted(glob.glob('./GenderData/ply_files/*.ply'))[25:31]
    pred_dir = sorted(glob.glob('./SegClusClass_preds/*'))

    validation_labels = []

    Confs = []
    for i, file in enumerate(pred_dir):
        cloud = read_ply(pred_dir[i])
        cloud_orig = read_ply(pred_dir_orig[i])
        true = cloud['class']
        preds = cloud_orig['preds']

        validation_labels += [true]
        Confs += [fast_confusion(true, preds, label_values)]
        print(preds.shape, true.shape)


    val_proportions = np.zeros(nc_model, dtype=np.float32)
    i = 0
    for label_value in label_values:
        if label_value not in ignored_labels:
            val_proportions[i] = np.sum([np.sum(labels == label_value)
                                        for labels in validation_labels])
            i += 1

    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)

    for l_ind, label_value in reversed(list(enumerate(label_values))):
        if label_value in ignored_labels:
            C = np.delete(C, l_ind, axis=0)
            C = np.delete(C, l_ind, axis=1)


    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)
    
    
    print("Vanilla Segmentation results::: ")
    s = '{:5.2f} | '.format(100 * mIoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    print(s + '\n')
    
    print("SegClusClass results::: ")
    
    s = '{:5.2f} | '.format(100 * mIoU_seg)
    for IoU in IoUs_seg:
        s += '{:5.2f} '.format(100 * IoU)
    print(s + '\n')
    

    # Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
    # for i, (preds, truth) in enumerate(zip(predictions, targets)):

    #     # Confusions
    #     Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)    

    # C = np.sum(np.stack(Confs), axis=0)

    # # Remove ignored labels from confusions
    # for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
    #     if label_value in test_loader.dataset.ignored_labels:
    #         C = np.delete(C, l_ind, axis=0)
    #         C = np.delete(C, l_ind, axis=1)

    # IoUs = IoU_from_confusions(C)
    # mIoU = np.mean(IoUs)
    # s = '{:5.2f} | '.format(100 * mIoU)
    # for IoU in IoUs:
    #     s += '{:5.2f} '.format(100 * IoU)
    # print('-' * len(s))
    # print(s)
    # print('-' * len(s) + '\n')
    
    
    
    
    
    
    
    
#     s = '{:5.2f} | '.format(100 * mIoU)
#     for IoU in IoUs:
#         s += '{:5.2f} '.format(100 * IoU)
#     print(s + '\n')
    

    # Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
    # for i, (preds, truth) in enumerate(zip(predictions, targets)):

    #     # Confusions
    #     Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)    

    # C = np.sum(np.stack(Confs), axis=0)

    # # Remove ignored labels from confusions
    # for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
    #     if label_value in test_loader.dataset.ignored_labels:
    #         C = np.delete(C, l_ind, axis=0)
    #         C = np.delete(C, l_ind, axis=1)

    # IoUs = IoU_from_confusions(C)
    # mIoU = np.mean(IoUs)
    # s = '{:5.2f} | '.format(100 * mIoU)
    # for IoU in IoUs:
    #     s += '{:5.2f} '.format(100 * IoU)
    # print('-' * len(s))
    # print(s)
    # print('-' * len(s) + '\n')
    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)
