'''Visualization.

'''

import logging

import imageio
import scipy
import matplotlib

matplotlib.use('Agg')
from matplotlib import pylab as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import visdom

import config
import exp
from viz_utils import tile_raster_images

logger = logging.getLogger('cortex.viz')
visualizer = None
config_font = None
_options = dict(use_tanh=False, quantized=False, img=None, label_names=None,
                is_caption=False, is_attribute=False)

CHARS = ['_', '\n', ' ', '!', '"', '%', '&', "'", '(', ')', ',', '-', '.', '/',
         '0', '1', '2', '3', '4', '5', '8', '9', ':', ';', '=', '?', '\\', '`',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '*', '*',
         '*']
CHAR_MAP = dict((i, CHARS[i]) for i in xrange(len(CHARS)))


def init():
    global visualizer, config_font
    if config.VIZ is not None and ('server' in config.VIZ.keys() or
                                           'port' in config.VIZ.keys()):
        server = config.VIZ.get('server', None)
        port = config.VIZ.get('port', 8097)
        visualizer = visdom.Visdom(server=server, port=port)
        logger.info('Using visdom server at {}({})'.format(server, port))
    else:
        visualizer = visdom.Visdom()
        logger.info('Using local visdom server')
    config_font = config.VIZ.get('font')


def setup(use_tanh=None, quantized=None, img=None, label_names=None,
          is_caption=False, is_attribute=False, char_map=None, name=None):
    global _options, CHAR_MAP
    if use_tanh is not None: _options['use_tanh'] = use_tanh
    if quantized is not None: _options['quantized'] = quantized
    if img is not None: _options['img'] = img
    if label_names is not None: _options['label_names'] = label_names
    _options['is_caption'] = is_caption
    _options['is_attribute'] = is_attribute
    if is_caption and is_attribute:
        raise ValueError('Cannot be both attribute and caption')
    if char_map is not None: CHAR_MAP = char_map


def dequantize(images):
    images = np.argmax(images, axis=1).astype('uint8')
    images_ = []
    for image in images:
        img2 = Image.fromarray(image)
        img2.putpalette(_options['img'].getpalette())
        img2 = img2.convert('RGB')
        images_.append(np.array(img2))
    images = np.array(images_).transpose(0, 3, 1, 2).astype(floatX) / 255.
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


def save_images(images, num_x, num_y, out_file=None, labels=None,
                max_samples=None, margin_x=5, margin_y=5, image_id=0,
                caption='', title=''):
    if max_samples is not None:
        images = images[:max_samples]
        if labels is not None:
            labels = labels[:max_samples]

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
        labels = np.argmax(labels, axis=-1)

    if _options['quantized']:
        images = dequantize(images)
    elif _options['use_tanh']:
        images = 0.5 * (images + 1.)

    images = images * 255.

    dim_c, dim_x, dim_y = images.shape[-3:]
    if dim_c == 1:
        arr = tile_raster_images(
            X=images, img_shape=(dim_x, dim_y), tile_shape=(num_x, num_y),
            tile_spacing=(margin_y, margin_x), bottom_margin=margin_y)
        fill = 255
    else:
        arrs = []
        for c in xrange(dim_c):
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
        try:
            font = ImageFont.truetype(config.font, 9)
        except:
            if config_font is None:
                raise ValueError('font must be added to config file in '
                                 '`viz`.')
            font = ImageFont.truetype(config_font, 9)

        idr = ImageDraw.Draw(im)
        for i, label in enumerate(labels):
            x_ = (i % num_x) * (dim_x + margin_x)
            y_ = (i // num_x) * (dim_y + margin_y) + dim_y
            if _options['is_caption']:
                l_ = ''.join([CHAR_MAP[j] for j in label
                              if CHAR_MAP[j] != '\n'])
                l__ = [CHAR_MAP[j] for j in label]
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
            try:
                idr.text((x_, y_), l_, fill=fill, font=font)
            except Exception as e:
                print l_
                print l__
                print len(l_)
                raise e

    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr.transpose(2, 0, 1)
    visualizer.image(arr, opts=dict(title=title, caption=caption),
                     win='image_{}'.format(image_id),
                     env=exp.NAME)

    if out_file:
        im.save(out_file)


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
    bins = np.linspace(np.min(np.array(scores.values())),
                       np.max(np.array(scores.values())), 100)
    plt.clf()
    for k, v in scores.items():
        plt.hist(v, bins, alpha=0.5, label=k)
    plt.legend(loc='upper right')
    plt.savefig(out_file)
    hists = tuple(np.histogram(v, bins=bins)[0] for v in scores.values())
    X = np.column_stack(hists)
    visualizer.stem(
        X=X, Y=np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(99)]),
        opts=dict(legend=['Real', 'Fake']), win='hist_{}'.format(hist_id),
        env=exp.NAME)

