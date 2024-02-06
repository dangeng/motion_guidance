from colorwheel import flow_to_image
from PIL import Image
import numpy as np

def get_translation(x1, y1, x2, y2):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates
    flow = np.zeros((512,512,2))
    flow[:,:,0] = x2 - x1
    flow[:,:,1] = y2 - y1
    
    # Get image and paste to background
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow


def get_rotation(x1, y1, x2, y2):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates
    theta = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 400 * np.pi
    coords = np.arange(0, 512)
    xx, yy = np.meshgrid(coords, coords)
    xx = xx - x1
    yy = yy - y1
    dx = np.cos(theta) * xx - np.sin(theta) * yy
    dy = np.sin(theta) * xx + np.cos(theta) * yy

    flow = np.zeros((512,512,2))
    flow[:,:,0] = dx
    flow[:,:,1] = dy
    
    # Get image and paste to background
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow

def get_scale(x1, y1, x2, y2):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates
    factor = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 100
    coords = np.arange(0, 512)
    xx, yy = np.meshgrid(coords, coords)
    xx = xx - x1
    yy = yy - y1

    flow = np.zeros((512,512,2))
    flow[:,:,0] = xx * (factor - 1)
    flow[:,:,1] = yy * (factor - 1)
    
    # Get image and paste to background
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow

def get_scale_1d(x1, y1, x2, y2):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates
    factor = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 100
    theta = np.arctan2(x2 - x1, y2 - y1)
    coords = np.arange(0, 512)
    u = np.array([y2-y1, -x2+x1])
    xx, yy = np.meshgrid(coords, coords)
    xx = xx - x1
    yy = yy - y1
    v = np.stack((xx, yy), axis=2)

    o = v - ((v @ u) / (u @ u))[:,:,None] * u

    flow = o * (factor - 1)
     
    # Get image and paste to background
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow

def get_bezier(x0, control_points):
    # Replace this function with your actual implementation
    # Example: Creating a simple image based on arrow coordinates

    # Sort by y-value
    control_points = sorted(control_points, key=lambda x: x[1])

    amps = np.array(control_points)[:, 0] / 100 - 1
    ys = np.array(control_points)[:, 1]

    xx = np.arange(0, 512)
    xx = xx - x0

    flow = np.zeros((512,512,2))

    flow[:ys[0],:,0] = xx * amps[0]
    flow[ys[-1]:,:,0] = xx * amps[-1]

    if len(control_points) >= 2:
        for i in range(len(control_points) - 1):
            for y in range(ys[i], ys[i+1]):
                t = (y - ys[i]) / (ys[i+1] - ys[i])   # btwn [0,1]
                factor = (amps[i+1] - amps[i]) * (1 - np.cos(t*3.141592)) / 2. + amps[i]
                flow[y,:,0] = xx * factor
     
    flow_im = flow_to_image(flow)
    flow_im = Image.fromarray(flow_im)
    return flow_im, flow

