import torch
import torch.nn as nn
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import io
import base64

app = Flask(__name__)
# Define the modified Generator class to match the state dict


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(params['nz'], 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, params['nc'], 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


# Default parameter values
params = {
    'nz': 100,
    # Size of the 2 latent vector (the input to the generator).
    # Number of channels in the training images. For colored images, this is 3.
    'nc': 3,
}
# Load the checkpoint file
state_dict = torch.load('gen.pth', map_location=torch.device('cpu'))
# Create an instance of the modified Generator
model = Generator()
# Load the state dict with modification
model_dict = model.state_dict()
state_dict = {k: v for k,
              v in state_dict['generator'].items() if k in model_dict}
model_dict.update(state_dict)
model.load_state_dict(model_dict)

# Generate fake images


def generate_images(nz):
    noise = torch.randn(nz, params['nz'], 1, 1, device=torch.device('cpu'))
    fake = model(noise).detach().cpu()
    fake_imgs = vutils.make_grid(fake, padding=2, normalize=True)
    return fake_imgs

# Define a Flask route to display the form and processed results


@app.route('/', methods=['GET', 'POST'])

def display_form():
    if request.method == 'POST':
        # Get user input from the form
        nz = int(request.form['nz'])
        # Generate fake images with user-specified parameters
        fake_imgs = generate_images(nz)
        # Save the generated image to a buffer
        buffer = io.BytesIO()
        # Define a function to create the plot
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Generated Images (nz={})".format(nz))
        plt.imshow(np.transpose(fake_imgs, (1, 2, 0)))
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        # Convert the image buffer to base64 string
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return render_template('index.html', image_base64=image_base64)
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run()
