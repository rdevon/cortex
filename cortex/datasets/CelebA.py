"""
TODO
"""
import torchvision
import logging
import os

LOGGER = logging.getLogger('cortex.CelebA')

class CelebA(torchvision.datasets.ImageFolder):

    url = "https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=1"
    filename = "img_align_celeba.zip"

    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        super(CelebA, self).__init__(root, transform, target_transform)

    def download(self):
        """
        TODO
        """
        import errno
        import zipfile
        from six.moves import urllib

        root = self.root
        url = self.url

        root = os.path.expanduser(root)
        fpath = os.path.join(root, self.filename)
        image_dir = os.path.join(root, 'images')

        try:
            os.makedirs(image_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                LOGGER.info('Dataset exists, not downloading.')
                return
            else:
                raise

        # downloads file
        if os.path.isfile(fpath):
            LOGGER.info('Using downloaded file: {}'.format(fpath))
        else:
            try:
                LOGGER.info('Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)
            except Exception as e:
                if url[:5] == 'https':
                    url = url.replace('https:', 'http:')
                    LOGGER.info(
                        'Failed download. Trying https -> http instead.'
                        ' Downloading ' + url + ' to ' + fpath)
                    urllib.request.urlretrieve(url, fpath)
                else:
                    raise

        zip_ref = zipfile.ZipFile(fpath, 'r')
        zip_ref.extractall(image_dir)
        zip_ref.close()
