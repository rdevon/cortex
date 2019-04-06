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
from .utils import convert_to_numpy, compute_tsne, print_hypers
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


_plotly_colors = [[31, 119, 180],  # muted blue
                  [255, 127, 14],  # safety orange
                  [44, 160, 44],  # cooked asparagus green
                  [214, 39, 40],  # brick red
                  [148, 103, 189],  # muted purple
                  [140, 86, 75],  # chestnut brown
                  [227, 119, 194],  # raspberry yogurt pink
                  [127, 127, 127],  # middle gray
                  [188, 189, 34],  # curry yellow-green
                  [23, 190, 207]  # blue-teal
                  ]


def setup(server=None, port=8097, font=None, update_frequency=0, plot_window=0,
          viz_off=False, viz_mode='visdom', plot_test_only=False, align_colors=False):
    '''

    Args:
        update_frequency (int): Frequency in data steps for displaying visualization.
        plot_window (int): Window for data steps for displaying plots.
        viz_off (bool): Turn off visualization.
        viz_mod (str): Visualizer mode. Only `visdom` supported but others coming soon.
        plot_test_only (bool): Show only plots for test set.
        align_colors (bool): Aligns train / test colors in plots.

    '''
    if viz_off:
        return

    global visualizer, config_font
    config_font = font

    if server and port:
        logger.info('Using visdom version {}'.format(visdom.__version__))
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

    if visualizer is not None:
        viz_handler.setup(update_frequency=update_frequency, plot_window=plot_window, viz_mode=viz_mode,
                          plot_test_only=plot_test_only, align_colors=align_colors)
        info = print_hypers(exp.ARGS, s='Model hyperparameters: ', visdom_mode=True)
        visualizer.text(info, env=exp.NAME, win='info')


class VizHandler():
    '''Hanldes all of your visualization needs.

    '''
    def __init__(self, ):
        self.clear()
        self.output_dirs = exp.OUT_DIRS
        self.prefix = exp._file_string('')
        self.image_scale = (-1, 1)
        self.last_step = -1
        self.stored_plots = dict(losses=dict(train=dict(), test=dict()),
                                 results=dict(train=dict(), test=dict()),
                                 times=dict(train=dict()),
                                 grads=dict(train=dict(), test=dict()))

    def setup(self, update_frequency, plot_window, viz_mode, plot_test_only, align_colors):
        '''Set up the handler.

        Args:
            update_frequency (int): Number of steps per update.
            plot_window (int): Window for plots.
            viz_mode (str): Mode for visualization. Only visdom support right now.
            plot_test_only (bool): Plot only eval values.
            align_colors (bool): Aligns train / test colors in plots.

        '''
        self.update_frequency = update_frequency
        self.plot_window = plot_window
        self.plot_test_only = plot_test_only
        self.align_colors = align_colors

    def clear(self):
        '''Clears visualizer.

        '''
        self.images = {}
        self.scatters = {}
        self.histograms = {}
        self.heatmaps = {}

    def update(self, viz_fn):
        '''Run a visualization update.

        This function will trigger visualization every N steps in training.

        Args:
            viz_fn: Visualization function from model.

        '''
        def show():
            viz_fn(auto_input=True)
            self.plot()
            self.clear()

        if self.update_frequency == 0:  # Update at the beginning of every epoch.
            current_step = exp.INFO['epoch']
            if current_step != self.last_step:
                show()
            self.last_step = exp.INFO['epoch']
        else:
            current_step = exp.INFO['data_steps']
            if ((current_step % self.update_frequency) == 0) and (current_step != self.last_step):
                show()
            self.last_step = exp.INFO['data_steps']

    def add_image(self, im, name='image', labels=None):
        '''Adds an image to the handler.

        '''
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
        '''Shows images.

        '''
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
        self.clear()

    def plot(self):
        """Updates the plots for the results.

        """

        def send_plot(k, X, Y, win, name, dash, linecolor, update=None):
            X = np.array(X).transpose()
            Y = np.array(Y).transpose()

            if self.plot_window == 0:
                label = 'Epochs'
            else:
                label = 'Updates'
            title_parts = k.split('.')
            title = '\n'.join(title_parts)
            ylabel = title_parts[-1]

            if linecolor is not None:
                linecolor = np.array([linecolor])

            opts = dict(
                xlabel=label,
                ylabel=ylabel,
                title=title,
                dash=dash,
                update=update,
                linecolor=linecolor)

            visualizer.line(Y=Y, X=X, env=exp.NAME, win=win, name=name, opts=opts, update=update)

        for result_type in ('times', 'losses', 'results', 'grads'):
            stored_plots = self.stored_plots[result_type]
            results = exp.RESULTS.results[result_type]

            modes = list(results.keys())

            # This option can make it so we only show eval results
            if self.plot_test_only:
                if 'test' in modes and result_type != 'times':
                    modes = ['test']
                elif 'train' in modes and result_type == 'times':
                    modes = ['train']
                else:
                    modes = []

            update = None
            seen_result_keys = set()

            for mode in modes:
                if mode not in stored_plots:
                    stored_plots[mode] = dict()

                stored_mode = stored_plots[mode]
                result = results[mode]
                result_keys = list(result.keys())

                lc_idx = 0
                for result_key in result_keys:
                    if result_key not in stored_mode:
                        stored_mode[result_key] = ([], [])

                    # Fetch result line not already pulled and combine with stored.
                    X, Y = stored_mode[result_key]

                    # Some chunks weren't finished when update called
                    if len(X) > 1 and self.plot_window != 0 and (X[-1] % self.plot_window) != 0:
                        start = X[-2] + 1
                    elif len(X) > 0:
                        start = X[-1] + 1
                    else:
                        start = 0
                    X_add, Y_add = exp.RESULTS.chunk(mode, result_type, result_key, start=start, window=self.plot_window)

                    # Add new chunks to storage
                    if len(X) > 1 and self.plot_window != 0 and (X[-2] % self.plot_window) != 0:
                        X = X[:-2] + X_add
                        Y = Y[:-2] + Y_add
                    else:
                        X = X + X_add
                        Y = Y + Y_add

                    # For some plots, there might not be a value for x = 0
                    if X[0] != 0:
                        X = [0] + X
                        Y = [Y[0]] + Y
                    stored_mode[result_key] = (X, Y)

                    # Results get their own window, losses etc shared window.
                    if result_type == 'results':
                        win = 'line_{}'.format(result_key)
                        name = mode
                        if result_key in seen_result_keys:
                            update = 'append'
                        else:
                            update = None
                        title = result_key
                        seen_result_keys.add(result_key)
                    else:
                        win = result_type
                        name = '{} ({})'.format(result_key, mode)
                        title = result_type
                        seen_result_keys.add(result_key)

                    if self.align_colors:
                        linecolor = _plotly_colors[lc_idx]
                    else:
                        linecolor = None

                    # Eval values get a dash line.
                    if mode == 'test' or self.plot_test_only:
                        dash = np.array(['dashdot'])
                    else:
                        dash = None

                    send_plot(title, X, Y, win, name, dash=dash, linecolor=linecolor, update=update)
                    update = 'append'

                    lc_idx += 1
                    lc_idx = lc_idx % len(_plotly_colors)


viz_handler = VizHandler()


def save_text(labels, out_file=None, text_id=0,
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
    '''

    Args:
        images:
        num_x:
        num_y:
        out_file:
        labels:
        max_samples:
        margin_x:
        margin_y:
        image_id:
        caption:
        title:

    Returns:

    '''
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
            dim_c, dim_x, dim_y = image.shape[-3:]
            image = image.reshape((num_x, num_y, dim_c, dim_x, dim_y))
            image = image.transpose(0, 3, 1, 4, 2)
            image = image.reshape(num_x * dim_x, num_y * dim_y, dim_c)
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
