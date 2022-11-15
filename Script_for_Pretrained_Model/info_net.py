from torchinfo import summary
model_path=r"F:\pix2pix_FC\checkpoints\pix2pix_FC\295_net_G.pth"
model = GAN_generator = torch.load(model_path, map_location='cpu').to('cpu')
batch_size = 1
summary(model, input_size=(batch_size, 1, 28, 28))