import os

import tensorflow as tf
import numpy as np
from PIL import Image

def random_position():
    return np.random.random([1, 100]) * 2.0 - 1.0
    #return (np.random.random([100]) * 2.0 - 1.0).resize((1, 100))

def make_movie(generator_model, output_folder, transition_time, frames):
    generator = tf.keras.models.load_model(generator_model)
    noise1 = random_position()
    noise2 = random_position()
    step = 0
    for frame in range(frames):
        t = float(frame % transition_time) / transition_time
        noise = noise1 * (1.0 - t) + noise2 * t
        noise.resize((1, 100))
        # print(frame, t, noise[0][:3])
        result = generator.predict(noise)[0]
        pixels = ((result + 1.0) * 127.0).astype('uint8')
        (w, h, _) = pixels.shape
        pixels.resize((w, h))
        #print(pixels.shape)
        img = Image.fromarray(pixels)
        filename = os.path.join(output_folder, 'img-{:04d}.png'.format(frame + 1))
        print(filename)
        img.save(filename)
        step += 1
        if step >= transition_time:
            step = 0
            noise1 = noise2
            noise2 = random_position()

if __name__ == '__main__':
    generator_model = 'generator.h5'
    output_folder = 'out/'
    transition_time = 100
    frames = 500
    make_movie(generator_model, output_folder, transition_time, frames)
