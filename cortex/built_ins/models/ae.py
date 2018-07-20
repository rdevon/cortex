# from cortex.plugins import ModelPlugin
# from cortex.main import run
# import torch.nn.functional as F
# from cortex.built_ins.models.image_coders import ImageEncoder, ImageDecoder
# from cortex.built_ins.networks.ae_network import AENetwork
#
#
# class AE(ModelPlugin):
#     plugin_name = 'AE'
#     defaults = dict(
#         data=dict(
#             batch_size=dict(train=64, test=64), inputs=dict(inputs='images')),
#         optimizer=dict(optimizer='Adam', learning_rate=1e-4),
#         train=dict(save_on_lowest='losses.ae'))
#
#     def __init__(self):
#         super().__init__()
#         self.encoder = ImageEncoder()
#         self.decoder = ImageDecoder()
#
#     def build(self, dim_z=64, dim_encoder_out=64):
#         self.encoder.build(dim_encoder_out)
#         self.decoder.build(dim_z)
#         encoder = self.nets.encoder
#         decoder = self.nets.decoder
#         ae = AENetwork(encoder, decoder)
#         self.nets.ae = ae
#
#     def routine(self, inputs, targets, ae_criterion=F.mse_loss):
#         ae = self.nets.ae
#         outputs = ae(inputs)
#         r_loss = ae_criterion(
#             outputs, inputs, size_average=False) / inputs.size(0)
#         self.losses.ae = r_loss
#
#     def visualize(self, inputs, targets):
#         ae = self.nets.ae
#         outputs = ae(inputs)
#         self.add_image(outputs, name='reconstruction')
#         self.add_image(inputs, name='ground truth')
#
#
# if __name__ == '__main__':
#     autoencoder = AE()
#     run(model=autoencoder)
