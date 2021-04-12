import numpy
import os
import re
import tensorflow as tf
import xml.etree.ElementTree as et
from glob import iglob


class ImageSet(tf.keras.utils.Sequence):
    def __init__(self, path_annotations, path_images, path_cache=None, batch_size=1, image_size=224, length=None, use_annotations=True):
        self.batch_size = batch_size
        self.image_size = image_size
        self.path = path_images
        self.class_names = None
        self.length = length

        self.list = []
        self.synset_map = {}     # synset -> its internal index
        synsets = []
        annotated = 0

        # check if the cache is here
        cache_sep = '***************'
        if path_cache and os.path.isfile(path_cache):
            with open(path_cache, 'r') as cache:
                # read image entries
                line = cache.readline()[:-1]
                while line:
                    if line == cache_sep: break
                    l = line.split(' ')
                    self.list.append((
                        int(l[0]),
                        ' '.join(l[3:]),
                        (float(l[1]), float(l[2]))
                    ))
                    line = cache.readline()[:-1]
                
                # read synsets
                line = cache.readline()[:-1]
                while line:
                    self.synset_map[line] = len(synsets)
                    synsets.append(line)
                    line = cache.readline()[:-1]
        else:
            # reopen the cache file
            if path_cache:
                cache = open(path_cache, 'w')

            for jpeg_filename in iglob(os.path.join(path_images, '**', '*.JPEG'), recursive=True):
                # check if annotation is avaliable
                filename = jpeg_filename[len(path_images) + 1 : -5]
                xml_filename = os.path.join(path_annotations, filename + '.xml')

                if os.path.isfile(xml_filename):
                    # proceed with the annotation
                    annotated += 1
                    xml = et.parse(xml_filename).getroot()
                    
                    # find biggest object
                    max_area = 0
                    for obj in xml.findall('object'):
                        bbox = obj.find('bndbox')
                        area = (int(bbox.find('xmax').text) - int(bbox.find('xmin').text)) * (int(bbox.find('ymax').text) - int(bbox.find('ymin').text))
                        assert area > 0, "Negative bounding box area got"
                        if area > max_area:
                            max_area = area
                            bestObj = obj

                    # get its synset
                    synset = bestObj.find('name').text[1:]

                    # get its center
                    if use_annotations:
                        bbox = bestObj.find('bndbox')
                        x, y = (int(bbox.find('xmin').text) + int(bbox.find('xmax').text)) // 2, (int(bbox.find('ymin').text) + int(bbox.find('ymax').text)) // 2
                        size = xml.find('size')
                        x /= float(size.find('width').text)
                        y /= float(size.find('height').text)
                    else:
                        x, y = 0.5, 0.5

                else:
                    # proceed without annotation
                    match = re.match('n(\d+)\%sn(\d+)_(\d+)' % (os.path.sep), filename)
                    assert match and match.group(1) == match.group(2), "Cannot match: " + filename
                    synset = match.group(1)
                    x, y = 0.5, 0.5

                # store
                if not synset in self.synset_map:
                    self.synset_map[synset] = len(self.synset_map)
                    synsets.append(synset)
                syn_idx = self.synset_map[synset]
                self.list.append((syn_idx, filename, (x,y)))
                
                # save to cache
                if path_cache:
                    # order: "synsetIndex centerX centerY filename"
                    cache.write('%d %0.8f %0.8f %s\n' % (syn_idx, x, y, filename))

                if len(self.list) % 100000 == 0:
                    print('  %dk images processed' % (len(self.list) / 1000))

            # store the synset list to cache
            if path_cache:
                cache.write(cache_sep + '\n')
                for synset in synsets:
                    cache.write(synset + '\n')
            print('%d images got, %0.2f%% annotated' % (len(self.list), 100 * annotated / len(self.list)))

        # remap classes indices
        yy = range(len(self.synset_map))
        xx = sorted(self.synset_map.keys())
        remap = dict(zip([self.synset_map[_] for _ in xx], yy))
        self.synset_map = dict(zip(xx, yy))
        for i in range(len(self.list)):
            entry = self.list[i]
            self.list[i] = (remap[entry[0]], entry[1], entry[2])


    def supply_class_names(self, filename):
        """ Loads class indexes to names map
        """
        with open(filename, 'r') as file:
            self.class_names = {}
            for line in file.readlines():
                entries = line.split(' ')
                self.class_names[self.synset_map[entries[0][1:]]] = entries[-1]


    def get_class_idx(self, i):
        """ Retrieves the class index of an image at position i
        """
        return self.list[i % len(self.list)][0]


    def get_class_name(self, i):
        """ Retrieves the class name of an image at position i
        """
        return self.class_names[self.get_class_idx(i)]


    def shuffle(self):
        import random
        random.shuffle(self.list)


    def filter(self, classes):
        """ Filters the dataset keeping only selected classes.
        """
        assert len(numpy.unique(classes)) == len(classes), "Expecting unique class numbers"
        filter_map = dict(zip(classes, range(len(classes))))

        old_synsets = list(self.synset_map.keys())
        new_synset_map = {}
        for s in old_synsets:
            if self.synset_map[s] in classes:
                new_synset_map[s] = filter_map[self.synset_map[s]]
        self.synset_map = new_synset_map

        if self.class_names:
            new_class_names = []
            for c in classes:
                new_class_names.append(self.class_names[c])
            self.class_names = new_class_names

        new_list = []
        for entry in self.list:
            if entry[0] in classes:
                new_list.append((filter_map[entry[0]],) + entry[1:])
        self.list = new_list


    def samples(self, class_idx):
        """ Generator producing samples of a specific class, for illustration purposes.
        """
        i = 0
        while True:
            syn_idx, filename, center = self.list[i]
            if syn_idx == class_idx:
                filename = os.path.join(self.path, filename + '.JPEG')
                with open(filename, 'rb') as file:
                    image = tf.io.decode_image(file.read(), channels=3)
                yield (image, self.get_class_name(i))
            i = (i + 1) % len(self.list)


    def numClasses(self):
        return len(self.synset_map)


    def _load(self, idx):
        """ Loads an image by its index
        """
        syn_idx, filename, center = self.list[idx % len(self.list)]
        filename = os.path.join(self.path, filename + '.JPEG')
        with open(filename, 'rb') as file:
            # load image
            image = tf.io.decode_image(file.read(), channels=3, dtype=tf.float32)
            
            # compute bbox
            h, w = image.shape[:2]
            c = min(w, h)
            box = [center[1] - 0.5*c/h, center[0] - 0.5*c/w, center[1] + 0.5*c/h, center[0] + 0.5*c/w]

            # shift the bbox to not to go out of the image
            if box[0] < 0: box[2] -= box[0]; box[0] = 0
            if box[1] < 0: box[3] -= box[1]; box[1] = 0
            if box[2] > 1: box[0] -= box[2] - 1; box[2] = 1
            if box[3] > 1: box[1] -= box[3] - 1; box[3] = 1

            # crop and resize
            return tf.image.crop_and_resize(tf.expand_dims(image, 0), [box], [0], (self.image_size, self.image_size))


    def __getitem__(self, i):
        i0, i1 = i * self.batch_size, min(len(self.list), (i + 1) * self.batch_size)
        batch = []
        labels = []
        assert i0 < len(self.list), "Requested batch index is out of range"
        for j in range(i0, i1):
            image = self._load(j)
            batch.append(image)
            labels.append(self.get_class_idx(j))
        return tf.concat(batch, 0), tf.one_hot(labels, self.numClasses())


    def __len__(self):
        return self.length or (len(self.list) + self.batch_size - 1) // self.batch_size


    def on_epoch_end(self):
        import random
        random.shuffle(self.list)


    def make_tfrecord(self, fname_pattern, num_files):
        """ Transforms the image set into a set of TF record files if not yet
            Args:
              fname_pattern:    TF record file name pattern.
              num_files:        Number of files expected.
                                The function quits immediately if the number of existing files matching the pattern is
                                equal to `num_files`.
        """
        filenames = tf.io.gfile.glob(fname_pattern)
        if len(filenames) == num_files:
            return
        n = 0
        for shard_idx in range(num_files):
            with tf.io.TFRecordWriter(fname_pattern.replace('*', '%02d' % shard_idx)) as writer:
                for i in range(shard_idx, len(self.list), num_files):
                    image, label = self._load(i), self.get_class_idx(i)
                    image = tf.cast(tf.clip_by_value(tf.squeeze(image) * 255, 0, 255), tf.uint8)
                    image_str = tf.io.encode_jpeg(image, quality=100, optimize_size=True, chroma_downsampling=False)
                    feature={
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_str.numpy()])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    }
                    ex = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(ex.SerializeToString())
                    n += 1
                    if n % 1000 == 0: print(n, "of", len(self.list), "examples written...")
