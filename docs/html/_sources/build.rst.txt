Custom demos
~~~~~~~~~~~~

While cortex has built-in functionality, but it is meant to meant to be
used with your own modules. An example of making a model that works with
cortex can be found at:
https://github.com/rdevon/cortex/blob/master/demos/demo_classifier.py
and https://github.com/rdevon/cortex/blob/master/demos/demo_custom_ae.py

Documentation on the API can be found here:
https://github.com/rdevon/cortex/blob/master/cortex/plugins.py

For instance, the demo autoencoder can be used as:

::

   python cortex/demos/demo_custom_ae.py --help

A walkthrough a custom classifier:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let’s look a little more closely at the autoencoder demo above to see
what’s going on. cortex relies on using and overriding methods of
plugins classes.

First, let’s look at the methods, ``build``, ``routine``, and
``visualize``. These are special methods for the plugin that can be
overridden to change the behavior of your model for your needs.

The signature of these functions look like:

::

       def build(self, dim_z=64, dim_encoder_out=64):
           ...

       def routine(self, inputs, targets, ae_criterion=F.mse_loss):
           ...

       def visualize(self, inputs, targets):
           ...

Each of these functions have arguments and keyword arguments. Note that
the keyword arguments showed up in the help in the above example. This
is part of the functionality of cortex: it manages your hyperparameters
to these functions, organizes them, and provides command line control
automatically. Even the docstrings are used in the command line, so
other users can get the usage docs directly from there.

The arguments are *data*, which are to be manipulated as needed in those
methods. These are for the most part handled automatically, but all of
these methods can be used as normal functions as well.

Building models
^^^^^^^^^^^^^^^

The ``build`` function takes the hyperparameters and sets networks.

::


   class Autoencoder(nn.Module):
       def __init__(self, encoder, decoder):
           super(Autoencoder, self).__init__()
           self.encoder = encoder
           self.decoder = decoder

       def forward(self, x, nonlinearity=None):
           encoded = self.encoder(x)
           decoded = self.decoder(encoded)
           return decoded

   ...

       def build(self, dim_z=64, dim_encoder_out=64):
           encoder = nn.Sequential(
               nn.Linear(28, 256),
               nn.ReLU(True),
               nn.Linear(256, 28),
               nn.ReLU(True))
           decoder = nn.Sequential(
               nn.Linear(28, 256),
               nn.ReLU(True),
               nn.Linear(256, 28),
               nn.Sigmoid())
           self.nets.ae = Autoencoder(encoder, decoder)

All that’s being done here is the hyperparameters are being used to
create an instance of an ``nn.Module`` subclass, which is being added to
the set of “nets”. Note that they keyword ``ae`` is very important, as
this is going to be how you retrieve your nets and define their losses
farther down.

Also note that cortex *only* currently supports ``nn.Module`` subclasses
from Pytorch.

Defining losses and results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Adding losses and results from your model is easy, just compute your
graph given you models and data, then add the losses and results by
setting those members:

::

       def routine(self, inputs, targets, ae_criterion=F.mse_loss):
           encoded = self.nets.ae.encoder(inputs)
           outputs = self.nets.ae.decoder(encoded)
           r_loss = ae_criterion(
               outputs, inputs, size_average=False) / inputs.size(0)
           self.losses.ae = r_loss

Additional results can be added similarly. For instance, in the demo
classifier:

::

       def routine(self, inputs, targets, criterion=nn.CrossEntropyLoss()):
           ...
           classifier = self.nets.classifier

           outputs = classifier(inputs)
           predicted = torch.max(F.log_softmax(outputs, dim=1).data, 1)[1]

           loss = criterion(outputs, targets)
           correct = 100. * predicted.eq(
               targets.data).cpu().sum() / targets.size(0)

           self.losses.classifier = loss
           self.results.accuracy = correct

Visualization
~~~~~~~~~~~~~

Cortex allows for visualization using visdom, and this can be defined in
a similar way as above:

::

       def visualize(self, images, inputs, targets):
           predicted = self.predict(inputs)
           self.add_image(images.data, labels=(targets.data, predicted.data),
                          name='gt_pred')

See the ModelPlugin API for more more details.

Putting it together
~~~~~~~~~~~~~~~~~~~

Finally, we can specify default arguments:

::

       defaults = dict(
           data=dict(
               batch_size=dict(train=64, test=64), inputs=dict(inputs='images')),
           optimizer=dict(optimizer='Adam', learning_rate=1e-4),
           train=dict(save_on_lowest='losses.ae'))

and then add ``cortex.main.run`` to ``__main__``:

::

   if __name__ == '__main__':
       autoencoder = AE()
       run(model=autoencoder)

And that’s it. cortex also allows for lower-level functions to be
overridden (e.g., train_step, eval_step, train_loop, etc) with more
customizability coming soon. For more examples of usage, see the
built-in models:
https://github.com/rdevon/cortex/tree/master/cortex/built_ins/models