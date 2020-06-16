#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir = '/scratch/users/suihong/training_data/'  # Training data path
result_dir = '/scratch/users/suihong/ProGAN_MultiChannel_Reusults_ConditionedtoMultiConditions_TF/'  # result data path

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.
tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
#tf_config['gpu_options.allow_growth']          = False     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
#env.CUDA_VISIBLE_DEVICES                       = '0'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '0'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
dataset     = EasyDict()                                    # Options for dataset.load_dataset().
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
grid        = EasyDict(size='6by8')                         # Options for train.setup_snapshot_image_grid().

dataset = EasyDict(tfrecord_dir= 'TrainingData(MultiChannels_Version4)')

desc += '-2gpu'; num_gpus = 2; sched.minibatch_base = 32; sched.minibatch_dict = {4: 32, 8: 32, 16: 32, 32: 32, 64: 32}; sched.G_lrate_dict = {4: 0.0025, 8: 0.005, 16: 0.005, 32: 0.0035, 64: 0.0025}; sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); train.total_kimg = 60000
sched.max_minibatch_per_gpu = {32: 32, 64: 32}

# ** Uncomment following one line of code if using conventional GAN training process. **
#desc += '-nogrowing'; sched.lod_initial_resolution = 64; train.total_kimg = 10000

# Disable individual features.
#desc += '-nopixelnorm'; G.use_pixelnorm = False
#desc += '-nowscale'; G.use_wscale = False; D.use_wscale = False
#desc += '-noleakyrelu'; G.use_leakyrelu = False
#desc += '-nosmoothing'; train.G_smoothing = 0.0
#desc += '-norepeat'; train.minibatch_repeats = 1
#desc += '-noreset'; train.reset_opt_for_new_lod = False

#----------------------------------------------------------------------------
# Utility scripts.
# Functions used to generate fake images, interpolation-videos
# To run, uncomment the appropriate line and launch train.py.

#train = EasyDict(func='util_scripts.generate_fake_images', run_id=60, num_pngs=1000); num_gpus = 1; desc = 'fake-images-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_fake_images', run_id=60, grid_size=[8,6], num_pngs=10, image_shrink=1); num_gpus = 1; desc = 'fake-grids-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_interpolation_video', run_id=2, grid_size=[8,6], duration_sec=40.0, smoothing_sec=1.0); num_gpus = 2; desc = 'interpolation-video-' + str(train.run_id)
#train = EasyDict(func='util_scripts.generate_training_video', run_id=60, duration_sec=20.0); num_gpus = 1; desc = 'training-video-' + str(train.run_id)

#----------------------------------------------------------------------------
# Utility scripts.
# Functions used for metric evaluation of swd and swd_distribution plot.
# To run, uncomment the appropriate line and launch train.py.

# multi-scale sliced wasserstein distance values for different networks during training
#train = EasyDict(func='util_scripts.evaluate_metrics', run_id=881, log='metric-swd-4k.txt', metrics=['swd'], num_images=30, real_passes=2); num_gpus = 1; desc = train.log.split('.')[0] + '-' + str(train.run_id)

#distribution of facies models based on multi-scale sliced wasserstein distance for different networks during training
#train = EasyDict(func='util_scripts.evaluate_metrics_swd_distributions', run_id=881, log='metric-swd_distri-300.txt', metrics=['swd_distri'], num_images_per_group = 20, num_groups=10, real_passes=1); num_gpus = 2; desc = train.log.split('.')[0] + '-' + str(train.run_id)

# distribution of facies models produced from conventionally and progressively trained generators at the same plot.
#network_dir_conv = '/scratch/users/suihong/ProGAN_MultiChannel_Reusults_ConditionedtoMultiConditions_TF/881-Unconditional_trad/network-snapshot-016640.pkl'
#network_dir_prog = '/scratch/users/suihong/ProGAN_MultiChannel_Reusults_ConditionedtoMultiConditions_TF/Unconditional_prog/network-snapshot-011520.pkl'
#train = EasyDict(func='util_scripts.evaluate_metrics_swd_distributions_training_trad_prog', run_id = 881, network_dir_conv = network_dir_conv, network_dir_prog = network_dir_prog, log='metric-swd_distri_training_trad_prog-40.txt', metrics=['swd_distri_training_trad_prog'], num_images_per_group = 50, num_groups=10, real_passes=1); num_gpus = 2; desc = train.log.split('.')[0]
#----------------------------------------------------------------------------
