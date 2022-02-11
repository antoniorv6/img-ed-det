import cv2
import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from Logger import *
import glob
import random
import distortions

dataset_path = {
    "gdl": "GDL/*",
    "imagenette": {
        "train": "imagenette2/*/n0[29,3]*/*.JPEG",
        "val": "imagenette2/*/n0[1]*/*.JPEG",
        "test": "imagenette2/*/n0[21]*/*.JPEG",
    },
}


class DataGenerator:
    def __init__(
        self,
        split='train',
        image_size=(224, 224),
        shuffle=True,
        val_size=0.1,
        test_size=0.1,
        dataset="gdl",
        batch_size=1,
        distortions_to_apply=distortions.no_rep_distortions,
        augmentations_to_apply=distortions.augment_distortions,
        min_n_distortions=1,
        max_n_distortions=1,
        reps_per_distortion=1,
        # n_distortions=None,
        min_pairs=1,
        max_pairs=1,
    ):
        assert dataset in dataset_path.keys()

        ds = dataset_path[dataset]
        if isinstance(ds, str):
            images = glob.glob(os.path.join("Data", ds))
            train_size = 1-(val_size+test_size)
            if split == 'train':
                images = images[:int(train_size*len(images))]
            elif split == 'val':
                images = images[int(train_size*len(images)):-int(test_size*len(images))]
            else:
                images = images[-int(test_size*len(images)):]
        else:
            images = glob.glob(os.path.join("Data", ds[split]))

        # self.images_list, self.validation_list = train_test_split(
        #     images, test_size=val_size, shuffle=True, random_state=1
        # )
        self.images_list = images
        self.image_size = image_size

        self.shuffle = shuffle
        self.images_index = 0

        self.min_n_distortions = min_n_distortions
        self.max_n_distortions = max_n_distortions
        self.reps_per_distortion = reps_per_distortion
        self.distortions = copy.copy(distortions_to_apply)
        self.augmentations = copy.copy(augmentations_to_apply)
        self.img_shape = self.image_size + (3,)
        self.min_pairs = min_pairs
        self.max_pairs = max_pairs
        self.batch_size = batch_size
        self.reset()

        print("dists", len(self.distortions))
        
    @classmethod
    def params_from_experiment(
        cls,
        head='fcn',
        n_samples='single',
        seq_len='single',
        dataset='imagenette',
        repetitions=True
    ):
        max_samples = 5
        max_reps_per_distortion = 2
        distortions_to_apply = distortions.rep_distortions if repetitions else distortions.no_rep_distortions
        max_distortions = len(distortions_to_apply)
        if repetitions:
            max_distortions *= max_reps_per_distortion
        
        seq_len = (1 if seq_len=='single' else max_distortions) if isinstance(seq_len, str) else seq_len

        datagen_params = dict(
            dataset=dataset,
            distortions_to_apply=distortions_to_apply,
            reps_per_distortion=max_reps_per_distortion,
            max_n_distortions=seq_len,
            max_pairs=1 if n_samples=='single' else max_samples,
        )
        return datagen_params

    def read_image(self, path):
        return cv2.resize(cv2.imread(path), self.image_size)

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.images_list)
            # np.random.shuffle(self.validation_list)
        self.it = iter(self.images_list)
        # self.val_iter = iter(self.validation_list)
    
    def next(self):
        try:
            v = next(self.it)
        except:
            self.reset()
            v = next(self.it)
        return v
    
    def get_random_edit_sequence(self, include_end_token=False):
        editions = np.random.choice(
            list(range(len(self.distortions))) * self.reps_per_distortion + ([len(self.distortions)] if include_end_token else []),
            np.random.randint(self.min_n_distortions, self.max_n_distortions+1+int(include_end_token)),
            replace=False
        )
        return editions

    def generate_image(self, n_images=None):
        images = [self.read_image(self.next()) for _ in range(n_images or np.random.randint(self.min_pairs, self.max_pairs+1))]
        source_images = images

        # First edition pass
        edition_process = np.random.binomial(1, 0.5, len(self.augmentations))
        edition_process = np.arange(len(edition_process))[
            edition_process.astype(np.bool)
        ]
        for i in edition_process:
            source_images = [getattr(distortions, self.augmentations[i])(im) for im in source_images]

        editions = self.get_random_edit_sequence()

        edited_images = source_images
        for edit in editions:
            edited_images = [getattr(distortions, self.distortions[edit])(img) for img in edited_images]

        X_source = np.stack(source_images, axis=0).astype(np.uint8)
        X_target = np.stack(edited_images, axis=0).astype(np.uint8)
        Y = editions.astype(np.int32)
        Y = np.append(Y, np.full(self.max_n_distortions - Y.shape[0] + 1, len(self.distortions)))

        return X_source, X_target, Y
    
    def old_generate_image(self, n_images=1):
        images = [self.read_image(self.next()) for _ in range(n_images)]
        source_images = images

        # First edition pass
        edition_process = np.random.binomial(1, 0.5, len(self.augmentations))
        edition_process = np.arange(len(edition_process))[
            edition_process.astype(np.bool)
        ]
        for i in edition_process:
            source_images = [getattr(distortions, self.augmentations[i])(im) for im in source_images]

        if self.mode == "single":
            edition_to_detect = np.zeros(len(self.distortions))
            edition_to_detect[random.randint(0, len(self.distortions) - 1)] = 1
        else:
            if self.n_distortions is None:
                edition_to_detect = np.zeros(len(self.distortions))
                edition_to_detect[
                    np.random.choice(
                        len(self.distortions), np.random.randint(1, len(self.distortions)+1), replace=False
                    )
                ] = 1
            else:
                edition_to_detect = np.zeros(len(self.distortions))
                edition_to_detect[
                    np.random.choice(
                        len(self.distortions), self.n_distortions, replace=False
                    )
                ] = 1

        edition_to_detect = np.arange(len(edition_to_detect))[
            edition_to_detect.astype(np.bool)
        ]
        if self.mode == "multiple_sorted" or "next":
            np.random.shuffle(edition_to_detect)

        edited_images = source_images
        for dist in edition_to_detect:
            edited_images = [getattr(distortions, self.distortions[dist])(img) for img in edited_images]

        X_source = np.stack(source_images, axis=0).astype(np.uint8)
        X_target = np.stack(edited_images, axis=0).astype(np.uint8)
        Y = edition_to_detect.astype(np.int32)

        if self.mode == "single" or self.mode == "multiple":
            label = np.zeros(len(self.distortions), dtype=np.int32)
            label[Y] = 1
        elif self.mode == "multiple_sorted":
            label = np.zeros(self.label_shape, dtype=np.int32)
            label[[np.arange(len(Y)), Y]] = 1
            label[len(Y) :, -1] = 1
        elif self.mode == 'next':
            label = np.zeros(self.label_shape, dtype=np.int32)
            label[Y[0]] = 1

        Y = label

        # print(X_source.shape)
        # print(X_target.shape)
        # print(Y.shape)

        return X_source, X_target, Y

    def generator(self):
        yield self.generate_image(n_images=np.random.randint(self.min_pairs, self.max_pairs+1))

    # def val_generator(self):
    #     yield self.generate_image(self.val_iter, n_images=np.random.randint(self.min_pairs, self.max_pairs+1))

    def batch_generator(self):
        n_images = np.random.randint(self.min_pairs, self.max_pairs+1)
        batch = [[], [], []]
        for _ in range(self.batch_size):
            batch_i = self.generate_image(n_images=n_images)
            for i in range(len(batch_i)):
                batch[i].append(batch_i[i])
        for i in range(len(batch)):
            batch[i] = np.stack(batch[i])
        #print(batch[0].shape)
        #print(len(batch))
        # quit()
        yield tuple(batch)


    # def val_batch_generator(self):
    #     n_images = np.random.randint(self.min_pairs, self.max_pairs+1)
    #     batch = [[], [], []]
    #     for _ in range(self.batch_size):
    #         batch_i = self.generate_image(self.val_iter, n_images=n_images)
    #         for i in range(len(batch_i)):
    #             batch[i].append(batch_i[i])
    #     for i in range(len(batch)):
    #         batch[i] = np.stack(batch[i])
    #     yield tuple(batch)

    # def get_val_size(self):
    #     return len(self.validation_list)

    def get_size(self):
        return len(self.images_list)

    # def get_val_steps(self):
    #     return len(self.validation_list) // self.batch_size

    def get_steps(self):
        return len(self.images_list) // self.batch_size


def main():
    CONST_TEST_FOLDER = "Test/Generator/"
    dataGen = DataGen()
    BATCH_SIZE = 16
    dataGen.reset()
    X_source = []
    X_target = []
    Y = []

    DATA_GEN_LOG_INFO("CREATING TEST FOLDER")
    try:
        os.mkdir(CONST_TEST_FOLDER)
        DATA_GEN_LOG_INFO("TEST FOLDER CREATED")
    except OSError:
        DATA_GEN_LOG_WARNING("TEST FOLDER ALREADY IN DIRECTORY")

    DATA_GEN_LOG_INFO(f"GENERATING {1157} SAMPLES")

    for elx in range(1157):
        X_source, X_target, Y = dataGen.train_batch(BATCH_SIZE)
        for i, element in enumerate(Y):
            # imageName = ' '.join([str(elem) for elem in element]) + f"_{i}"
            for idxed, bit in enumerate(element):
                if bit == 1.0:
                    break

            cv2.imwrite(
                CONST_TEST_FOLDER
                + str(elx + i)
                + "_"
                + bitwise_names[idxed]
                + "_src.jpg",
                X_source[i],
            )
            cv2.imwrite(
                CONST_TEST_FOLDER
                + str(elx + i)
                + "_"
                + bitwise_names[idxed]
                + "_tar.jpg",
                X_target[i],
            )

    DATA_GEN_LOG_INFO(f"IMAGES GENERATED AND WRITTEN IN {CONST_TEST_FOLDER}")

    pass


if __name__ == "__main__":
    main()
#
