v1 = {
    "resblock": "1",
    "num_gpus": 0,
    "batch_size": 16,
    "learning_rate": 0.0004,
    "adam_b1": 0.8,
    "adam_b2": 0.99,
    "lr_decay": 0.999,
    "seed": 1234,

    "upsample_rates": [8,8,2,2],
    "upsample_kernel_sizes": [16,16,4,4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3,7,11],
    "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
    "resblock_initial_channel": 256,

    "segment_size": 8192,
    "num_mels": 80,
    "num_freq": 1025,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,

    "sampling_rate": 22050,

    "fmin": 0,
    "fmax": 8000,
    "fmax_loss": None,

    "num_workers": 4,

    "dist_config": {
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:54321",
        "world_size": 1
    }
}
