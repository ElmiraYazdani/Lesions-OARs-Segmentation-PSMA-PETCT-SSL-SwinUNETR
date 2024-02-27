# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    SmartCacheDataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


def get_loader(args):
    num_workers = 4

    train_list = load_decathlon_datalist(
        args.json_list, False, "training", base_dir=args.data_dir
    )
    val_list = load_decathlon_datalist(
        args.json_list, False, "validation", base_dir=args.data_dir
    )

    print(
        "total number of trainin data: {}".format(len(train_list))
    )
    print(
        "total number of validation data: {}".format(len(val_list))
    )
    transforms_list = [
        LoadImaged(keys=["image"],image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        SpatialPadd(
            keys="image",
            spatial_size=[args.roi_x, args.roi_y, args.roi_z],
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            k_divisible=[args.roi_x, args.roi_y, args.roi_z],
        ),
        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=[args.roi_x, args.roi_y, args.roi_z],
            num_samples=args.sw_batch_size,
            random_center=True,
            random_size=False,
        ),
        ToTensord(keys=["image"]),
    ]

    train_transforms = Compose(transforms_list)
    val_transforms = Compose(transforms_list)
    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(
            data=train_list,
            transform=train_transforms,
            cache_rate=0.5,
            num_workers=num_workers,
        )
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=train_list,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = Dataset(
            data=train_list, transform=train_transforms
        )

    if args.distributed:
        train_sampler = DistributedSampler(
            dataset=train_ds, even_divisible=True, shuffle=True
        )
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        drop_last=False,
    )
    # sample = next(iter(train_loader))

    val_ds = Dataset(data=val_list, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader
