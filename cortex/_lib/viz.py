"""
Visualization.
"""
import logging
from os import path

import imageio
import numpy as np
from PIL import Image, ImageDraw
import visdom

from . import data, exp
from .utils import convert_to_numpy, compute_tsne
from .viz_utils import tile_raster_images
import matplotlib
import subprocess
from cortex._lib.config import _yes_no

matplotlib.use('Agg')
from matplotlib import pylab as plt  # noqa E402

__author__ = 'R Devon Hjelm'
__author_email__ = 'erroneus@gmail.com'

logger = logging.getLogger('cortex.viz')
config_font = None
visualizer = None
_options = dict(use_tanh=False, quantized=False, img=None, label_names=None,
                is_caption=False, is_attribute=False)

CHARS = ['_', '\n', ' ', '!', '"', '%', '&', "'", '(', ')', ',', '-', '.', '/',
         '0', '1', '2', '3', '4', '5', '8', '9', ':', ';', '=', '?', '\\', '`',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '*', '*',
         '*']
CHAR_MAP = dict((i, CHARS[i]) for i in range(len(CHARS)))


def init(viz_config):
    global visualizer, config_font, viz_process
    if viz_config is not None and ('server' in viz_config.keys() or
                                   'port' in viz_config.keys()):
        server = viz_config.get('server', None)
        port = viz_config.get('port', 8097)
        visualizer = visdom.Visdom(server=server, port=port)
        if not visualizer.check_connection():
            if _yes_no("No Visdom server runnning on the configured address. "
                       "Do you want to start it?"):
                viz_bash_command = "python -m visdom.server"
                viz_process = subprocess.Popen(viz_bash_command.split())
                logger.info('Using visdom server at {}({})'.format(server, port))
            else:
                visualizer = None
    else:
        if _yes_no("Visdom configuration is not specified. Please run 'cortex setup' "
                   "to configure Visdom server. Do you want to continue with "
                   "the default address ? (localhost:8097)"):
            viz_bash_command = "python -m visdom.server"
            viz_process = subprocess.Popen(viz_bash_command.split())
            visualizer = visdom.Visdom()
            logger.info('Using local visdom server')
        else:
            visualizer = None
    config_font = viz_config.get('font')


def setup(use_tanh=None, quantized=None, img=None, label_names=None,
          is_caption=False, is_attribute=False, char_map=None, name=None):
    global _options, CHAR_MAP
    if use_tanh is not None:
        _options['use_tanh'] = use_tanh
    if quantized is not None:
        _options['quantized'] = quantized
    if img is not None:
        _options['img'] = img
    if label_names is not None:
        _options['label_names'] = label_names
    _options['is_caption'] = is_caption
    _options['is_attribute'] = is_attribute
    if is_caption and is_attribute:
        raise ValueError('Cannot be both attribute and caption')
    if char_map is not None:
        CHAR_MAP = char_map


class VizHandler():
    def __init__(self):
        self.clear()
        self.output_dirs = exp.OUT_DIRS
        self.prefix = exp._file_string('')
        self.image_scale = (-1, 1)

    def clear(self):
        self.images = {}
        self.scatters = {}
        self.histograms = {}
        self.heatmaps = {}

    def add_image(self, im, name='image', labels=None):
        im = convert_to_numpy(im)
        mi, ma = self.image_scale
        im = (im - mi) / float(ma - mi)
        if labels is not None:
            labels = convert_to_numpy(labels)
        if name in self.images:
            logger.warning('{} already added to '
                           'visualization. Use the name kwarg'
                           .format(name))
        self.images[name] = (im, labels)

    def add_histogram(self, hist, name='histogram'):
        if name in self.histograms:
            logger.warning('{} already added'
                           ' to visualization.'
                           ' Use the name kwarg'
                           .format(name))
        hist = convert_to_numpy(hist)
        self.histograms[name] = hist

    def add_heatmap(self, hm, name='heatmap'):
        if name in self.heatmaps:
            logger.warning('{} already'
                           ' added to visualization.'
                           ' Use the name kwarg'
                           .format(name))
        hm = convert_to_numpy(hm)
        self.heatmaps[name] = hm

    def add_scatter(self, sc, labels=None, name='scatter'):
        sc = convert_to_numpy(sc)
        labels = convert_to_numpy(labels)

        self.scatters[name] = (sc, labels)

    def show(self):
        image_dir = self.output_dirs['image_dir']
        for i, (k, (im, labels)) in enumerate(self.images.items()):
            if image_dir:
                logger.debug('Saving images to {}'.format(image_dir))
                out_path = path.join(
                    image_dir, '{}_{}_image.png'.format(self.prefix, k))
            else:
                out_path = None

            save_images(im, 8, 8, out_file=out_path, labels=labels,
                        max_samples=64, image_id=1 + i, caption=k)

        for i, (k, (sc, labels)) in enumerate(self.scatters.items()):

            if sc.shape[1] == 1:
                raise ValueError('1D-scatter not supported')
            elif sc.shape[1] > 2:
                logger.info('Scatter greater than 2D. Performing TSNE to 2D')
                sc = compute_tsne(sc)

            if image_dir:
                logger.debug('Saving scatter to {}'.format(image_dir))
                out_path = path.join(
                    image_dir, '{}_{}_scatter.png'.format(self.prefix, k))
            else:
                out_path = None

            save_scatter(sc, out_file=out_path,
                         labels=labels, image_id=i,
                         title=k)

        for i, (k, hist) in enumerate(self.histograms.items()):
            if image_dir:
                logger.debug('Saving histograms to {}'.format(image_dir))
                out_path = path.join(
                    image_dir, '{}_{}_histogram.png'.format(self.prefix, k))
            else:
                out_path = None
            save_hist(hist, out_file=out_path, hist_id=i)

        for i, (k, hm) in enumerate(self.heatmaps.items()):
            if image_dir:
                logger.debug('Saving heatmap to {}'.format(image_dir))
                out_path = path.join(
                    image_dir, '{}_{}_heatmap.png'.format(self.prefix, k))
            else:
                out_path = None
            save_heatmap(hm, out_file=out_path, image_id=i, title=k)


def plot(epoch, init=False):
    '''Updates the plots for the reults.

    Takes the last value from the summary and appends this to the visdom plot.

    '''
    def get_X_Y_legend(key, v_train, v_test):
        min_e = max(0, epoch - 1)
        if init:
            min_e = 0
        if min_e == epoch:
            Y = [[v_train[0], v_train[0]]]
        else:
            Y = [v_train[min_e:epoch + 1]]
        legend = []

        if v_test is not None:
            if min_e == epoch:
                Y.append([v_test[0], v_test[0]])
            else:
                Y.append(v_test[min_e:epoch + 1])

            if min_e == epoch:
                X = [[-1, 0], [-1, 0]]
            else:
                X = [range(min_e, epoch + 1), range(min_e, epoch + 1)]

            legend.append('{} (train)'.format(key))
            legend.append('{} (test)'.format(key))
        else:
            legend.append(key)

            if min_e == epoch:
                X = [[-1, 0]]
            else:
                X = [range(min_e, epoch + 1)]

        return X, Y, legend

    train_summary = exp.SUMMARY['train']
    test_summary = exp.SUMMARY['test']
    for k in train_summary.keys():
        v_train = train_summary[k]
        v_test = test_summary[k] if k in test_summary.keys() else None

        if isinstance(v_train, dict):
            Y = []
            X = []
            legend = []
            for k_ in v_train:
                vt = v_test.get(k_) if v_test is not None else None
                X_, Y_, legend_ = get_X_Y_legend(k_, v_train[k_], vt)
                Y += Y_
                X += X_
                legend += legend_
        else:
            X, Y, legend = get_X_Y_legend(k, v_train, v_test)

        opts = dict(
            xlabel='epochs',
            legend=legend,
            ylabel=k,
            title=k)

        X = np.array(X).transpose()
        Y = np.array(Y).transpose()

        if init:
            update = None
        else:
            update = 'append'

        visualizer.line(
            Y=Y,
            X=X,
            env=exp.NAME,
            opts=opts,
            win='line_{}'.format(k),
            update=update)


def dequantize(images):
    images = np.argmax(images, axis=1).astype('uint8')
    images_ = []
    for image in images:
        img2 = Image.fromarray(image)
        img2.putpalette(_options['img'].getpalette())
        img2 = img2.convert('RGB')
        images_.append(np.array(img2))
    images = np.array(images_).transpose(0, 3, 1, 2) / 255.
    return images


def save_text(labels, max_samples=64, out_file=None, text_id=0,
              caption=''):
    labels = np.argmax(labels, axis=-1)
    char_map = _options['label_names']
    l_ = [''.join([char_map[j] for j in label]) for label in labels]

    logger.info('{}: {}'.format(caption, l_[0]))
    visualizer.text('\n'.join(l_), env=exp.NAME,
                    win='text_{}'.format(text_id))

    if out_file is not None:
        with open(out_file, 'w') as f:
            for l__ in l_:
                f.write(l__)


def save_images(images, num_x, num_y, out_file=None, labels=None,  # noqa C901
                max_samples=None, margin_x=5, margin_y=5, image_id=0,
                caption='', title=''):
    if labels is not None:
        if isinstance(labels, (tuple, list)):
            labels = zip(*labels)
    if max_samples is not None:
        images = images[:max_samples]

    if labels is not None:
        if _options['is_caption']:
            margin_x = 80
            margin_y = 80
        elif _options['is_attribute']:
            margin_x = 25
            margin_y = 200
        elif _options['label_names'] is not None:
            margin_x = 20
            margin_y = 25
        else:
            margin_x = 5
            margin_y = 12

    if _options['quantized']:
        images = dequantize(images)

    images = images * 255.

    dim_c, dim_x, dim_y = images.shape[-3:]
    if dim_c == 1:
        arr = tile_raster_images(
            X=images, img_shape=(dim_x, dim_y), tile_shape=(num_x, num_y),
            tile_spacing=(margin_y, margin_x), bottom_margin=margin_y)
        fill = 255
    else:
        arrs = []
        for c in range(dim_c):
            arr = tile_raster_images(
                X=images[:, c].copy(), img_shape=(dim_x, dim_y),
                tile_shape=(num_x, num_y),
                tile_spacing=(margin_y, margin_x),
                bottom_margin=margin_y, right_margin=margin_x)
            arrs.append(arr)

        arr = np.array(arrs).transpose(1, 2, 0)
        fill = (255, 255, 255)

    im = Image.fromarray(arr)

    if labels is not None:
        idr = ImageDraw.Draw(im)
        for i, label in enumerate(labels):
            x_ = (i % num_x) * (dim_x + margin_x)
            y_ = (i // num_x) * (dim_y + margin_y) + dim_y
            if _options['is_caption']:
                l_ = ''.join([CHAR_MAP[j] for j in label
                              if CHAR_MAP[j] != '\n'])
                # l__ = [CHAR_MAP[j] for j in label]
                l_ = l_.strip()
                if len(l_) == 0:
                    l_ = '<EMPTY>'
                if len(l_) > 30:
                    l_ = '\n'.join(
                        [l_[x:x + 30] for x in range(0, len(l_), 30)])
            elif _options['is_attribute']:
                attribs = [j for j, a in enumerate(label) if a == 1]
                l_ = '\n'.join(_options['label_names'][a] for a in attribs)
            elif _options['label_names'] is not None:
                l_ = _options['label_names'][label]
                l_ = l_.replace('_', '\n')
            else:
                l_ = str(label)
            idr.text((x_, y_), l_, fill=fill)

    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)
    visualizer.image(arr, opts=dict(title=title, caption=caption),
                     win='image_{}'.format(image_id),
                     env=exp.NAME)

    if out_file:
        im.save(out_file)


def save_heatmap(X, out_file=None, caption='', title='', image_id=0):
    visualizer.heatmap(
        X=X,
        opts=dict(
            title=title,
            caption=caption),
        win='heatmap_{}'.format(image_id),
        env=exp.NAME)


def save_scatter(points, out_file=None, labels=None, caption='', title='',
                 image_id=0):
    if labels is not None:
        Y = (labels + 1.5).astype(int)
    else:
        Y = None

    names = data.DATA_HANDLER.get_label_names()
    Y = Y - min(Y) + 1
    if len(names) != max(Y):
        names = ['{}'.format(i + 1) for i in range(max(Y))]

    visualizer.scatter(
        X=points,
        Y=Y,
        opts=dict(
            title=title,
            caption=caption,
            legend=names,
            markersize=5),
        win='scatter_{}'.format(image_id),
        env=exp.NAME)


def save_movie(images, num_x, num_y, out_file=None, movie_id=0):
    if out_file is None:
        logger.warning('`out_file` not provided. Not saving.')
    else:
        images_ = []
        for i, image in enumerate(images):
            if _options['quantized']:
                image = dequantize(image)
            dim_c, dim_x, dim_y = image.shape[-3:]
            image = image.reshape((num_x, num_y, dim_c, dim_x, dim_y))
            image = image.transpose(0, 3, 1, 4, 2)
            image = image.reshape(num_x * dim_x, num_y * dim_y, dim_c)
            if _options['use_tanh']:
                image = 0.5 * (image + 1.)
            images_.append(image)
        imageio.mimsave(out_file, images_)

    visualizer.video(videofile=out_file, env=exp.NAME,
                     win='movie_{}'.format(movie_id))


def save_hist(scores, out_file, hist_id=0):
    s = list(scores.values())
    bins = np.linspace(np.min(np.array(s)),
                       np.max(np.array(s)), 100)
    plt.clf()
    for k, v in scores.items():
        plt.hist(v, bins, alpha=0.5, label=k)
    plt.legend(loc='upper right')
    if out_file:
        plt.savefig(out_file)
    hists = tuple(np.histogram(v, bins=bins)[0] for v in s)
    X = np.column_stack(hists)
    visualizer.stem(
        X=X, Y=np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(99)]),
        opts=dict(legend=['Real', 'Fake']), win='hist_{}'.format(hist_id),
        env=exp.NAME)
